import torch
import cv2
import torchvision.transforms as ttf
import torch.nn as nn


class MyData(torch.utils.data.Dataset):
    def __init__(self, path_im, path_mask, transform):
        self.images = []
        self.masks = []
        self.transform = transform
        
        image = cv2.imread(path_im)
        mask = cv2.imread(path_mask, 0).reshape((1026, 1282, 1))
        
        self.images.append(image)
        self.masks.append(mask)
        
        
    def __getitem__(self, idx):
        image = self.images[idx] # (1026, 1282, 3)
        image = self.transform(image) # torch.Size([3, 1026, 1282])
        mask = self.masks[idx] # (1026, 1282)
        mask = self.transform(mask) # torch.Size([1, 1026, 1282])  # if not reshaped, totensor can also handle it
        return image, mask
    
    def __len__(self):
        assert len(self.images) == len(self.masks)
        return len(self.masks)
    

train_data = MyData("deeplab1.png", "deeplab2.png", ttf.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=False)
model.cuda()

epochs = 2
l_r = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=l_r)


for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    
    for input, mask in train_loader:
        input = input.cuda()
        mask = mask.cuda()
        
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        
    total_loss += loss.item()
    print('epoch: {} loss: {:0.4f}'.format(epoch + 1, total_loss / len(train_loader)))
    
    


