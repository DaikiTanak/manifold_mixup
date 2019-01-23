import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# Shake-shake implementation from https://github.com/owruby/shake-shake_pytorch/blob/master/models/shakeshake.py
class ShakeShake(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2, training=True):
        if training:
            alpha = torch.FloatTensor(x1.size(0)).uniform_().to("cuda:0")
            alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x1)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        beta = torch.FloatTensor(grad_output.size(0)).uniform_().to("cuda:0")
        beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
        # beta = Variable(beta)

        return beta * grad_output, (1 - beta) * grad_output, None

# SELayer
# https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, shake_shake=False):
        super(SEBasicBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        self.reduction = reduction

        self.shake_shake = shake_shake

        # bn - 3*3 conv - bn - relu - dropout - 3*3 conv - bn - add
        # https://arxiv.org/pdf/1610.02915.pdf
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.drop = nn.Dropout2d(p=0.3)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn3 = nn.BatchNorm2d(planes)

        if shake_shake:
            self.branch1 = self._make_branch(inplanes, planes, stride)
            self.branch2 = self._make_branch(inplanes, planes, stride)


    def _make_branch(self, inplanes, planes, stride=1):
        # bn - 3*3 conv - bn - relu - dropout - 3*3 conv - bn - add
        return nn.Sequential(
                nn.BatchNorm2d(inplanes),
                conv3x3(inplanes, planes, stride),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=False),
                nn.Dropout2d(p=0.3),
                conv3x3(planes, planes, stride),
                nn.BatchNorm2d(planes),
                SELayer(planes, self.reduction))


    def forward(self, x):

        residual = x

        if not self.shake_shake:

            # bn - 3*3 conv - bn - relu - dropout - 3*3 conv - bn - add
            out = self.bn1(x)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.drop(out)
            out = self.conv2(out)
            out = self.bn3(out)
            out = self.se(out)
            #######
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)

        if self.shake_shake:
            h1 = self.branch1(x)
            h2 = self.branch2(x)
            out = ShakeShake.apply(h1, h2, self.training)
            assert h1.size() == out.size()
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, shake_shake=False):
        super(SEBottleneck, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

        # bn - 1*1conv - bn - relu - 3*3conv - bn - relu - 1*1conv - bn
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn3 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * 4)

    def forward(self, x):
        residual = x
        # bn - 1*1conv - bn - relu - 3*3conv - bn - relu - 1*1conv - bn
        # This architecture is proposed in Deep Pyramidal Residual Networks.

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn4(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet18(num_classes, if_mixup=False, if_shake_shake=False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, mixup_hidden=if_mixup, shake_shake=if_shake_shake)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet34(num_classes, if_mixup=False):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, mixup_hidden=if_mixup)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet50(num_classes, if_mixup=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, mixup_hidden=if_mixup)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet101(num_classes, if_mixup=False):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes, mixup_hidden=if_mixup)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def se_resnet152(num_classes, if_mixup=False, if_shake_shake=False):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes, mixup_hidden=if_mixup, shake_shake=if_shake_shake)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model



class ResNet(nn.Module):
    # This ResNet does Manifold-Mixup.
    # https://arxiv.org/pdf/1806.05236.pdf
    def __init__(self, block, layers, num_classes=2, zero_init_residual=True, mixup_hidden=False, shake_shake=False):
        super(ResNet, self).__init__()
        self.mixup_hidden = mixup_hidden
        self.shake_shake = shake_shake
        self.inplanes = 64
        self.num_classes = num_classes
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        widen_factor = 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64*widen_factor, layers[0])
        self.layer2 = self._make_layer(block, 128*widen_factor, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256*widen_factor, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512*widen_factor, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion * widen_factor, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Heの初期化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual and (not shake_shake):
            for m in self.modules():
                if isinstance(m, SEBottleneck):
                    nn.init.constant_(m.bn4.weight, 0)
                elif isinstance(m, SEBasicBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if self.shake_shake:
                layers.append(block(self.inplanes, planes, shake_shake=True))
            else:
                layers.append(block(self.inplanes, planes, shake_shake=False))

        return nn.Sequential(*layers)

    def forward(self, x, lam=None, target=None):
        def mixup_process(out, target_reweighted, lam):
            # target_reweighted is one-hot vector
            # target is the taerget class.

            # shuffle indices of mini-batch
            indices = np.random.permutation(out.size(0))

            out = out*lam.expand_as(out) + out[indices]*(1-lam.expand_as(out))
            target_shuffled_onehot = target_reweighted[indices]
            target_reweighted = target_reweighted * lam.expand_as(target_reweighted) + target_shuffled_onehot * (1 - lam.expand_as(target_reweighted))
            return out, target_reweighted

        def to_one_hot(inp, num_classes):
            y_onehot = torch.FloatTensor(inp.size(0), num_classes)
            y_onehot.zero_()
            y_onehot.scatter_(1, inp.unsqueeze(1).cpu(), 1)
            return y_onehot.to("cuda:0")

        if self.mixup_hidden:
            layer_mix = np.random.randint(0,3)
        else:
            layer_mix = 0

        out = x

        if lam is not None:
            target_reweighted = to_one_hot(target, self.num_classes)

        if lam is not None and self.mixup_hidden and layer_mix == 0:
            out, target_reweighted = mixup_process(out, target_reweighted, lam)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        if lam is not None and self.mixup_hidden and layer_mix == 1:
            out, target_reweighted = mixup_process(out, target_reweighted, lam)

        out = self.layer2(out)
        if lam is not None and self.mixup_hidden and layer_mix == 2:
            out, target_reweighted = mixup_process(out, target_reweighted, lam)


        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        if lam is None:
            return out
        else:
            return out, target_reweighted
