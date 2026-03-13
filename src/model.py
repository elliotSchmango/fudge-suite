import torch
import torch.nn as nn
import torch.nn.functional as f

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #wider convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(f.relu(self.bn1(self.conv1(x))))
        x = self.pool(f.relu(self.bn2(self.conv2(x))))
        x = self.pool(f.relu(self.bn3(self.conv3(x))))
        
        x = torch.flatten(x, 1)
        
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x