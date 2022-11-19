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
train_loader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)
val_data = MyDataset(data, label)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=20, shuffle=False)


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
model.cuda()
print(model)
        
# hyperparams
epochs = 20
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(epochs):
    '''
        train
    '''
    model.train()
    total_loss = 0.0
    num_correct = 0
    
    for input, label in train_loader:
        optimizer.zero_grad()
        input = input.cuda()
        label = label.cuda()
        output = model(input)
        
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss
    
    epoch_loss = total_loss / len(train_loader)
    print('epoch: {} loss: {:0.4f}'.format(epoch + 1, epoch_loss))
    
    '''
        evaluate
    '''
    model.eval()
    num_correct = 0
    
    for input, label in val_loader:
        input = input.cuda()
        label = label.cuda()
        
        with torch.no_grad():
            output = model(input)
        pred = torch.argmax(output, 1)
        num_correct += ((pred == label).sum()).item()
    
    val_acc = 100 * num_correct / len(val_data)
    print('val_acc: {}'.format(val_acc))

  
'''
    inference
'''
test_data = torch.tensor([1.0, 2.0])
test_data = test_data.unsqueeze(0)
test_data = test_data.cuda()

with torch.no_grad():
    output = model(test_data)

pred = torch.argmax(output, 1)
print("Prediction:", pred)