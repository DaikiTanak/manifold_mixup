import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from squeeze_excitation import SELayer

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DualPathBlock(nn.Module):
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, increase, Groups, _type='normal', if_selayer=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.increase = increase

        if _type is 'proj':
            key_stride = 1
            self.has_proj = True
        if _type is 'down':
            key_stride = 2
            self.has_proj = True
        if _type is 'normal':
            key_stride = 1
            self.has_proj = False

        if self.has_proj:
            self.c1x1_w = self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_c+2*increase, kernel_size=1, stride=key_stride)

        if not if_selayer:
            self.layers = nn.Sequential(OrderedDict([
                ('c1x1_a', self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)),
                ('c3x3_b', self.BN_ReLU_Conv(in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=key_stride, padding=1, groups=Groups)),
                ('c1x1_c', self.BN_ReLU_Conv(in_chs=num_3x3_b, out_chs=num_1x1_c+increase, kernel_size=1, stride=1))
            ]))
        else:
            self.layers = nn.Sequential(OrderedDict([
                ('c1x1_a', self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)),
                ('c3x3_b', self.BN_ReLU_Conv(in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=key_stride, padding=1, groups=Groups)),
                ('c1x1_c', self.BN_ReLU_Conv(in_chs=num_3x3_b, out_chs=num_1x1_c+increase, kernel_size=1, stride=1)),
                ('se_layer', SELayer(num_1x1_c+increase))
            ]))

    def BN_ReLU_Conv(self, in_chs, out_chs, kernel_size, stride, padding=0, groups=1):
        return nn.Sequential(OrderedDict([
            ('norm', nn.BatchNorm2d(in_chs)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)),
        ]))

    def forward(self, x):
        data_in = torch.cat(x, dim=1) if isinstance(x, list) else x

        if self.has_proj:
            data_o = self.c1x1_w(data_in)
            data_o1 = data_o[:, :self.num_1x1_c, :, :]
            data_o2 = data_o[:, self.num_1x1_c:, :, :]
        else:
            data_o1 = x[0]
            data_o2 = x[1]

        out = self.layers(data_in)

        summ = data_o1 + out[:, :self.num_1x1_c, :, :]
        dense = torch.cat([data_o2, out[:, self.num_1x1_c:, :, :]], dim=1)

        return [summ, dense]


class DPN(nn.Module):

    def __init__(self, num_init_features=64, k_R=96, G=32,
                 k_sec=(3, 4, 20, 3), inc_sec=(16,32,24,128) #DPN-92
                 , num_classes=2, if_selayer=False, if_mixup=False):

        super(DPN, self).__init__()

        self.mixup_hidden=if_mixup
        self.num_classes = num_classes
        blocks = OrderedDict()

        # conv1
        blocks['conv1'] = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # conv2
        bw = 256
        inc = inc_sec[0]
        R = int((k_R*bw)/256)
        blocks['conv2_1'] = DualPathBlock(num_init_features, R, R, bw, inc, G, 'proj', if_selayer=if_selayer)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0]+1):
            blocks['conv2_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal', if_selayer=if_selayer)
            in_chs += inc

        # conv3
        bw = 512
        inc = inc_sec[1]
        R = int((k_R*bw)/256)
        blocks['conv3_1'] = DualPathBlock(in_chs, R, R, bw, inc, G, 'down', if_selayer=if_selayer)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1]+1):
            blocks['conv3_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal', if_selayer=if_selayer)
            in_chs += inc


        # conv4
        bw = 1024
        inc = inc_sec[2]
        R = int((k_R*bw)/256)
        blocks['conv4_1'] = DualPathBlock(in_chs, R, R, bw, inc, G, 'down', if_selayer=if_selayer)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2]+1):
            blocks['conv4_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal', if_selayer=if_selayer)
            in_chs += inc


        # conv5
        bw = 2048
        inc = inc_sec[3]
        R = int((k_R*bw)/256)
        blocks['conv5_1'] = DualPathBlock(in_chs, R, R, bw, inc, G, 'down', if_selayer=if_selayer)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3]+1):
            blocks['conv5_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal', if_selayer=if_selayer)
            in_chs += inc

        self.conv2_block = nn.Sequential()
        for i in range(1, k_sec[0]+1):
            self.conv2_block.add_module("conv2_{}".format(i), blocks['conv2_{}'.format(i)])
        self.conv3_block = nn.Sequential()
        for i in range(1, k_sec[1]+1):
            self.conv3_block.add_module("conv3_{}".format(i), blocks['conv3_{}'.format(i)])
        self.conv4_block = nn.Sequential()
        for i in range(1, k_sec[2]+1):
            self.conv4_block.add_module("conv4_{}".format(i), blocks['conv4_{}'.format(i)])
        self.conv5_block = nn.Sequential()
        for i in range(1, k_sec[3]+1):
            self.conv5_block.add_module("conv5_{}".format(i), blocks['conv5_{}'.format(i)])


        self.features = nn.Sequential(blocks)
        self.classifier = nn.Linear(in_chs, num_classes)


    def forward(self, x, lam=None, target=None):
        def mixup_process(out, target_reweighted, lam):
            # target_reweighted is one-hot vector
            # target is the taerget class.
            if isinstance(out, list):
                threshold = out[0].size(1)
                out = torch.cat(out, dim=1)

            # shuffle indices of mini-batch
            indices = np.random.permutation(out.size(0))

            out = out*lam.expand_as(out) + out[indices]*(1-lam.expand_as(out))
            target_shuffled_onehot = target_reweighted[indices]
            target_reweighted = target_reweighted * lam.expand_as(target_reweighted) + target_shuffled_onehot * (1 - lam.expand_as(target_reweighted))

            if isinstance(out, list):
                out = [out[:, :threshold, :, :], out[:, threshold:, :, :]]

            return out, target_reweighted

        def to_one_hot(inp, num_classes):
            y_onehot = torch.FloatTensor(inp.size(0), num_classes)
            y_onehot.zero_()
            y_onehot.scatter_(1, inp.unsqueeze(1).cpu(), 1)
            return y_onehot.to("cuda:0")


        if lam is None:
            features = torch.cat(self.features(x), dim=1)
            out = F.avg_pool2d(features, kernel_size=7).view(features.size(0), -1)
            out = self.classifier(out)
            return out

        else:

            layer_mix = np.random.randint(0,4)

            if lam is not None:
                target_reweighted = to_one_hot(target, self.num_classes)

            out = x

            if lam is not None and layer_mix == 0:
                out, target_reweighted = mixup_process(out, target_reweighted, lam)

            out = self.features.conv1(out)
            out = self.conv2_block(out)
            if lam is not None and layer_mix == 1:
                out, target_reweighted = mixup_process(out, target_reweighted, lam)

            out = self.conv3_block(out)
            if lam is not None and layer_mix == 2:
                out, target_reweighted = mixup_process(out, target_reweighted, lam)

            out = self.conv4_block(out)
            if lam is not None and layer_mix == 3:
                out, target_reweighted = mixup_process(out, target_reweighted, lam)

            out = self.conv5_block(out)
            features = torch.cat(out, dim=1)

            out = F.avg_pool2d(features, kernel_size=7).view(features.size(0), -1)
            out = self.classifier(out)
            return out, target_reweighted



def dpn92(num_classes=2, if_selayer=False, if_mixup=False):
    return DPN(num_init_features=64, k_R=96, G=32, k_sec=(3,4,20,3), inc_sec=(16,32,24,128), num_classes=num_classes,
               if_selayer=if_selayer, if_mixup=if_mixup)


def dpn98(num_classes=2, if_selayer=False, if_mixup=False):
    return DPN(num_init_features=96, k_R=160, G=40, k_sec=(3,6,20,3), inc_sec=(16,32,32,128), num_classes=num_classes,
                if_selayer=if_selayer, if_mixup=if_mixup)


def dpn131(num_classes=2, if_selayer=False, if_mixup=False):
    return DPN(num_init_features=128, k_R=160, G=40, k_sec=(4,8,28,3), inc_sec=(16,32,32,128), num_classes=num_classes,
                if_selayer=if_selayer, if_mixup=if_mixup)


def dpn107(num_classes=2, if_selayer=False, if_mixup=False):
    return DPN(num_init_features=128, k_R=200, G=50, k_sec=(4,8,20,3), inc_sec=(20,64,64,128), num_classes=num_classes,
                if_selayer=if_selayer, if_mixup=if_mixup)
