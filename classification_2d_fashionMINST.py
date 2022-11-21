import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as ttf
import pandas as pd
import numpy as np

'''
    DataLoader
'''
class MyData(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        df = pd.read_csv(path)
        self.transform = transform
        self.images = df.iloc[:, 1:].values.astype(np.uint8) # csv.values  astype(np.uint8)
        self.labels = df.iloc[:, 0].values
        
    def __len__(self):
        assert len(self.labels) == len(self.images)
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long) # torch.long
        
        image = self.images[idx].reshape(28, 28, 1) # shape (28, 28, 1)
        image = self.transform(image) # shape torch.Size([1, 28, 28])
        return image, label
        
        
train_data = MyData("data/fashion-mnist_test.csv", ttf.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
val_data = MyData("data/fashion-mnist_test.csv", ttf.ToTensor())
val_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=False)


'''
    model - ResNet18
'''
# model = torchvision.models.resnet18(pretrained=False)
# model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
# model.cuda()
# print(model)


'''
    model - self-defined
'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = Net()
model.cuda()


'''
    hyperparams
'''
epochs = 5
l_r = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=l_r)


'''
    training
'''
print("start training!\n")

for epoch in range(epochs):
    '''
        train
    '''
    model.train()
    total_loss = 0.0
    
    for input, label in train_loader:
        input, label = input.cuda(), label.cuda()
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
    num_correct = 0
    total_loss = 0.0
    
    for input, label in val_loader:
        input, label = input.cuda(), label.cuda()
        with torch.no_grad():
            output = model(input)
        loss = criterion(output, label)
        total_loss += loss.item()
        
        pred = torch.argmax(output, 1)
        num_correct += torch.sum(pred == label).item()
    print('validation loss: {} val_acc: {}'.format(total_loss / len(val_loader), num_correct / len(val_data)))
        