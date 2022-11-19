import torch
import torch.nn as nn
import torch.nn.functional as F


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        
        with open(path, 'r') as f:
            lines = f.readlines()
        
        self.data = []
        for line in lines:
            line = line.strip('\n')
            p1, p2, label = line.split(' ')
            self.data.append([float(p1), float(p2), int(label)])
            
    def __getitem__(self, ind):
        
        p1, p2, label = self.data[ind]
        return torch.tensor([p1, p2]), torch.tensor(label)
    
    def __len__(self):
        return len(self.data)
    

train_data = MyDataset("data_gender_train.txt")
train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
print(next(iter(train_loader)))

val_data = MyDataset("data_gender_val.txt")
val_loader = torch.utils.data.DataLoader(val_data, batch_size=10, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 2)
        self.bn1 = nn.BatchNorm1d(50)
        
    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        return self.fc2(x)

model = Net()
model.cuda()
print(model)


'''
model = nn.Sequential(
    nn.Linear(2, 50),
    nn.BatchNorm1d(50),
    nn.ReLU(),
    
    nn.Linear(50, 2),
)

model.cuda()
print(model)
'''


# hyperparameters
epochs = 20
l_r = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=l_r)


for epoch in range(epochs):
    '''
        train
    '''
    model.train()
    total_loss = 0.0
    
    for input, label in train_loader:
        input = input.cuda()
        label = label.cuda()
        
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print('epoch: {} loss: {:0.4f}'.format(epoch + 1, total_loss / len(train_loader)))
    
    '''
        validation
    '''
    model.eval()
    total_loss = 0.0
    num_correct = 0
    
    for input, label in val_loader:
        input, label = input.cuda(), label.cuda()
        
        with torch.no_grad():
            output = model(input)
        
        loss = criterion(output, label)
        total_loss += loss.item()
        pred = torch.argmax(output, 1)
        num_correct += (torch.sum(pred == label)).item()
        
    print('validation: loss: {:0.4f} acc: {:0.4f}%'.format(total_loss / len(val_loader), 100 * num_correct / len(val_data)))
    
    
'''
    inference
'''
test_data = torch.tensor([[1.82, 80.0]])

model.eval()
with torch.no_grad():
    input = test_data.cuda()
output = model(input)
pred = torch.argmax(output, 1).item()
print('Prediciton', pred)
        
        