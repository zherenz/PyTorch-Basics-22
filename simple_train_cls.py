import torch
import torch.nn as nn
import torch.nn.functional as F

x0=torch.randn(100,2)
x1=-torch.randn(100,2)
y0=torch.zeros(100)
y1=torch.ones(100)
data=torch.cat([x0,x1],dim=0).type(torch.FloatTensor) #shape(200,2)
label=torch.cat([y0,y1],dim=0).type(torch.LongTensor) #shape(200)

"""
    write your code here
"""

model = nn.Sequential(
    nn.Linear(2, 100),
    nn.BatchNorm1d(100),
    nn.ReLU(),
    
    nn.Linear(100, 50),
    nn.BatchNorm1d(50),
    nn.ReLU(),
    
    nn.Linear(50, 2)
)

model.cuda()
print(model)

# hyperparameters
epochs = 20
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print('start training!')
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    data, label = data.cuda(), label.cuda()
    
    output = model(data)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
    
    preds = torch.argmax(output, 1)
    num_correct = int(torch.sum(preds == label))
    # schedular.step()
    
    print('epoch: {} loss: {:0.4f} train_acc: {}'.format(epoch + 1, float(loss), 100 * num_correct / 200))
print("finish training!")
  