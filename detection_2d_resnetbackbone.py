import torch
import torch.nn as nn
import torchvision

train_loader = None

# return bbox and class
class ObjectDetector(nn.Module):
	def __init__(self, baseModel, numClasses):
		super().__init__()
		self.baseModel = baseModel
		self.numClasses = numClasses
		
        # build the regressor head for outputting the bounding box
		self.regressor = nn.Sequential(
			nn.Linear(baseModel.fc.in_features, 128),
			nn.ReLU(),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 4),
			nn.Sigmoid()
		)
		# build the classifier head to predict the class labels
		self.classifier = nn.Sequential(
			nn.Linear(baseModel.fc.in_features, 512),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(512, self.numClasses)
		)
		# in 512, out 512
		self.baseModel.fc = nn.Identity()
   	
	def forward(self, x):
		features = self.baseModel(x)
		bboxes = self.regressor(features)
		classLogits = self.classifier(features)
		# return the outputs as a tuple
		return bboxes, classLogits


base_model = torchvision.models.resnet18(pretrained=True)
model = ObjectDetector(base_model, 70)
classLossFunc = nn.CrossEntropyLoss()
bboxLossFunc = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5
BBOX, LABELS = 0.5, 0.5


for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    num_correct = 0
    
    for (images, labels, bboxes) in train_loader:
            images, labels, bboxes = images.cuda(), labels.cuda(), bboxes.cuda()
            optimizer.zero_grad()
            
            out_bboxes, out_labels = model(images)
            bboxLoss = bboxLossFunc(out_bboxes, bboxes)
            classLoss = classLossFunc(out_labels, labels)
            loss = (BBOX * bboxLoss) + (LABELS * classLoss)
            loss.backward()
            optimizer.step()

            total_loss += loss
            num_correct += (out_labels.argmax(1) == labels).type(torch.float).sum().item()