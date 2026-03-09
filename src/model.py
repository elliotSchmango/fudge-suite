import torch #importing torch libraries
import torch.nn as nn
import torch.nn.functional as f

##config neural network class for CIFAR-10
class Net(nn.Module):
    ##initialize network layers
    def __init__(self):
        super(Net, self).__init__()

        #convolutional layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        #fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    ##execute forward pass
    def forward(self, x):
        #convolutional layers
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        
        #fully connected layers
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        #output final predictions
        x = self.fc3(x)
        return x