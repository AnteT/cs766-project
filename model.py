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
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64*38*38)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x) 

class FakeFaceDetectorDevelopment(nn.Module):
    """NOTE: Best performing architecture. Facial image binary classifier with feature extract resizing, images in batch must be size ``(160, 160)``."""
    def __init__(self, d_input:int=32, d_output:int=64): # d_input = 32, d_output = 64 best results so far
        super(FakeFaceDetectorDevelopment, self).__init__()
        self.d_input, self.d_output = d_input, d_output
        self.conv1 = nn.Conv2d(3, d_input, 3, 1)
        self.conv2 = nn.Conv2d(d_input, 128, 3, 1)  # Increased output channels
        self.fc1 = nn.Linear(128*38*38, d_output)  # Adjusted input size
        self.fc2 = nn.Linear(d_output, 1)  # Adjusted input size

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 128*38*38)  # Adjusted to match new channel size
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

def count_model_params(model: nn.Module, separate:bool=False) -> int|tuple[int, int]:
    """
    Counts the number of parameters in the provided ``model`` and returns them. To count weights and biases separately, use ``separate=True`` to return a tuple of ``(num_weights, num_biases)``.

    Parameters:
        - ``model`` (nn.Module): The torch model to use for counting parameters.
        - ``separate`` (bool, optional): If the weights and biases should be counted and returned separately.

    Returns:
        ``int | tuple[int, int]``: The number of parameters or a tuple with the format ``(num_weights, num_biases)``.

    Note:
        - Only those parameters with the attribute ``requires_grad=True`` are included in the resulting counts.
    """
    if not separate:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    weights = sum(p.numel() for p in model.parameters() if p.requires_grad and len(p.size()) > 1)
    biases = sum(p.numel() for p in model.parameters() if p.requires_grad and len(p.size()) == 1)
    return weights, biases
    
if __name__ == '__main__':
    ffd = FakeFaceDetectorFE()
    num_params = count_model_params(ffd, separate=False)
    print(f'{num_params = }')
    ffd = FakeFaceDetector() # 1625056
    num_params = count_model_params(ffd, separate=False)
    print(f'{num_params = }')
    ffd = FakeFaceDetectorDevelopment() # 1625056
    num_params = count_model_params(ffd, separate=False)
    print(f'{num_params = }')    