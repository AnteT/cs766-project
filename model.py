import torch
import torch.nn as nn
import torch.nn.functional as F

class FakeFaceDetector(nn.Module):
    """Facial image binary classifier, images in batch must be size ``(64, 64)``."""
    def __init__(self, d_input:int=32):
        super(FakeFaceDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, d_input, 3, 1)
        self.conv2 = nn.Conv2d(d_input, 64, 3, 1)
        self.fc1 = nn.Linear(64*14*14, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1,64*14*14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)
    
class FakeFaceDetectorFE(nn.Module):
    """Facial image binary classifier with feature extract resizing, images in batch must be size ``(160, 160)``."""
    def __init__(self, d_input:int=32):
        super(FakeFaceDetectorFE, self).__init__()
        self.conv1 = nn.Conv2d(3, d_input, 3, 1)
        self.conv2 = nn.Conv2d(d_input, 64, 3, 1)
        self.fc1 = nn.Linear(64*38*38, 128)  # Changed input size
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        x = F.leaky_relu(self.conv1(x)) # testing leaky relu
        x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv2(x))
        x = F.leaky_relu(self.conv2(x)) # testing leaky relu
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64*38*38)  # Changed size
        # x = F.relu(self.fc1(x))
        x = F.leaky_relu(self.fc1(x)) # testing leaky relu
        x = self.fc2(x)
        return torch.sigmoid(x) 