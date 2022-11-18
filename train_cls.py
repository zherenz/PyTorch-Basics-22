import torch
import torch.nn as nn
import torch.nn.functional as F

x0=torch.randn(100,2)
x1=-torch.randn(100,2)
y0=torch.zeros(100)
y1=torch.ones(100)
data=torch.cat([x0,x1],dim=0).type(torch.FloatTensor) #shape(200,2)
label=torch.cat([y0,y1],dim=0).type(torch.LongTensor) #shape(200)

# data[0] = tensor([-2.2313, -0.5952])

"""
    write your code here
"""

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    def __getitem__(self, ind):
        x = self.data[ind]
        y = self.label[ind]
        return x, y
        
    def __len__(self):
        return len(self.data)        


train_data = MyDataset(data, label)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

model = Net()
print(model)
        
# hyperparams
epochs = 50
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_correct_train = 0
    
    for input, label in train_loader:
        optimizer.zero_grad()
        output = model(input)
        
        pred = torch.argmax(output, 1)
        num_correct = int(torch.sum(pred == label))
        
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss
    
    epoch_loss = total_loss / len(train_loader)
    train_acc = num_correct / len(train_data) * 100
    print('epoch: {} loss: {} train_acc: {}'.format(epoch + 1, epoch_loss, train_acc))




# print('start training!')
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#     data, label = data.cuda(), label.cuda()
    
#     output = model(data)
#     loss = criterion(output, label)
#     loss.backward()
#     optimizer.step()
    
#     preds = torch.argmax(output, 1)
#     num_correct = int(torch.sum(preds == label))
#     # schedular.step()
    
#     print('epoch: {} loss: {:0.4f} train_acc: {}'.format(epoch + 1, float(loss), 100 * num_correct / 200))
# print("finish training!")
  