import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import MTCNN

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
        self.conv1 = nn.Conv2d(3, d_input, kernel_size=(3,3), stride=(1,1))
        self.conv2 = nn.Conv2d(d_input, 128, kernel_size=(3,3), stride=(1,1))  # Increased output channels
        self.fc1 = nn.Linear(128*38*38, d_output)  # Adjusted input size
        self.fc2 = nn.Linear(d_output, 1)  # Adjusted input size

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)  # Adjusted to match new channel size
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
    
def print_model_structure(model: nn.Module) -> None:
    """Prints out the named parameters of the provided ``model`` in the format of ``layer: size`` and returns ``None``."""
    print(model)
    print("Model Structure:")
    for name, param in model.named_parameters():
        print(f"Layer: {name}, Size: {param.size()}  ({get_hwc_from_cwh(param)})")

def get_hwc_from_cwh(x:torch.Tensor) -> tuple[int, int, int]:
    """Extracts the size and converts the format from ``Channels, Width, Height`` to ``Height, Width, Channels`` and returns as a tuple."""
    c, w, h = '_', '_', '_'
    x = x.size()
    xdim = len(x)
    if xdim == 4:
        c, w, h = x[1], x[2], x[3]
    elif xdim == 3:
        c, w, h = x[0], x[1], x[2]
    elif xdim == 2:
        w, h = x[0], x[1]
    else:
        h = x[0]
    return (h, w, c)

if __name__ == '__main__':
    ffd = FakeFaceDetectorDevelopment(d_input=48, d_output=64) # 1625056
    num_params = count_model_params(ffd, separate=False)
    print(f'{num_params = }') 
    print_model_structure(ffd)
