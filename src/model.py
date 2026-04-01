import torch
import torch.nn as nn
import torchvision.models as models

class Net(nn.Module): #resnet18 architecture
    def __init__(self):
        super(Net, self).__init__()
        
        #load bare resnet18
        self.model = models.resnet18(weights=None)
        
        #match conv layers to CIFAR-10 (32x32x3)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        #remove max pooling layer
        self.model.maxpool = nn.Identity()
        
        #fc layers for 10 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.model(x)