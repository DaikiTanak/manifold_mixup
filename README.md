# manifold_mixup
pytorch implementation of manifold-mixup : https://arxiv.org/abs/1806.05236

This repo includes DenseNet (https://arxiv.org/pdf/1608.06993.pdf), ResNet (https://arxiv.org/abs/1512.03385), and Dual Path Networks (https://arxiv.org/pdf/1707.01629.pdf).


## How to train model?

### You first define dataset.
```
train = Dataset(X, y)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
```
### Then define the model.
- If you wanna use manifold mixup, define model as follows. (let parameter:if_mixup be True)  
```
model = densenet121(if_mixup=True)
```
or,  
```
model = se_resnet18(if_mixup=True)
```
or,
```
model = dpn98(if_mixup=True)
```
- Otherwise, let parameter:if_mixup be False.

### Start training.
``` python
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, dampening=0, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60, 80], gamma=0.1)

# Define beta distribution
def mixup_data(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam

loss_function = nn.CrossEntropyLoss()
bce_loss = torch.nn.BCELoss()

for epoch in range(EPOCH):
  scheduler.step()
  # Training Phase
  model.train()
  train_loss = 0
  for i, train_data in enumerate(tqdm(train_loader)):
      inputs, labels = train_data
      inputs = inputs.to(device)
      labels = labels.to(device)
      if not args.mixup:
          # if you don't use manifold mixup
          outputs = model(inputs)
          loss = loss_function(outputs, labels)

      elif args.mixup:
          # if you use manifold mixup
          lam = mixup_data(alpha=args.mixup_alpha)
          lam = torch.from_numpy(np.array([lam]).astype('float32')).to(device)
          output, reweighted_target = model(inputs, lam=lam, target=labels)
          loss = bce_loss(softmax(output), reweighted_target)

      train_loss += loss.item()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```
