import torch
import urllib
from PIL import Image
import torchvision.transforms as ttf
import cv2

'''
    data
'''
# ToTensor: Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
# to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 

input_image = Image.open("deeplab1.png").convert("RGB") # Image.open().convert("RGB")
# <PIL.Image.Image image mode=RGB size=1282x1026 at 0x1232024D4F0>

input_image = cv2.imread("deeplab1.png") # (1026, 1282, 3)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) # (1026, 1282, 3)

input_mask = cv2.imread("deeplab2.png", 0) # (1036, 1278)
# flag = 0, 8位深度，1通道
# flag = 1, 8位深度，3通道

transform = ttf.ToTensor()
input = transform(input_image) # torch.Size([3, 1026, 1282])
input = input.unsqueeze(0) # torch.Size([1, 3, 1026, 1282])
input = input.cuda()

mask = transform(input_mask) # torch.Size([1, 1036, 1278])
mask = mask.unsqueeze(0) # torch.Size([1, 1, 1036, 1278])
mask = mask.cuda()


'''
    model
'''
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.cuda()
# lossFunc = BCEWithLogitsLoss()


'''
    inference
'''
model.eval()
with torch.no_grad():
    output = model(input)['out'][0] # only one image
    pred = torch.argmax(output, 0) # no batch dim
# pred.shape torch.Size([1026, 1282])
