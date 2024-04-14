import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
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

class FakeFaceDetectorDevelopmentDebug(nn.Module):
    """Debug version of best performing model to output tensor shapes after each layer."""
    def __init__(self, d_input:int=32, d_output:int=64): # d_input = 32, d_output = 64 best results so far
        super(FakeFaceDetectorDevelopmentDebug, self).__init__()
        self.d_input, self.d_output = d_input, d_output
        self.conv1 = nn.Conv2d(3, d_input, kernel_size=(3,3), stride=(1,1))
        self.conv2 = nn.Conv2d(d_input, 128, kernel_size=(3,3), stride=(1,1))  # Increased output channels
        self.fc1 = nn.Linear(128*38*38, d_output)  # Adjusted input size
        self.fc2 = nn.Linear(d_output, 1)  # Adjusted input size

    def forward(self, x:torch.Tensor):
        print(f"On input:    {x.size()}")
        x = self.conv1(x)
        print(f"After conv1: {x.size()}")
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        print(f"After maxp1: {x.size()}")
        x = self.conv2(x)
        print(f"After conv2: {x.size()}")
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        print(f"After maxp2: {x.size()}")
        x = x.view(x.size(0), -1)  # Adjusted to match new channel size
        print(f"After flat:  {x.size()}")
        x = self.fc1(x)
        print(f"After fc1:   {x.size()}")
        x = F.leaky_relu(x)
        x = self.fc2(x)
        print(f"After fc2:   {x.size()}")
        return torch.sigmoid(x)
    
class FFXPhase(v2.Transform):
    """Facial Feature Extraction phase of 2-phase model architecture, overrides torch.Transform to provide fallback transformation and normalization if face not detected and extracted."""
    def __init__(self, fail_thresholds:tuple[int,int,int]=[0.6, 0.7, 0.7]) -> None:
        super(FFXPhase, self).__init__()
        self.fail_thresholds = fail_thresholds
        # Primary preferential transform
        self.pt = v2.Compose([
             MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=self.fail_thresholds, factor=0.709, post_process=False)
            ,v2.ToDtype(torch.float32, scale=True)
            ,v2.Resize(size=(160,160), antialias=True)                                            
            ,v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Secondary failsafe transform
        self.st = v2.Compose([
             v2.ToImage()
            ,v2.ToDtype(torch.float32, scale=True)
            ,v2.Resize(size=(160,160), antialias=True)
            ,v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ,v2.ToPILImage()
        ])
    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        """Overrides call method ensuring a correctly processed tensor is returned in either outcome of phase-1 of FFX + FFD."""
        xtp = self.pt(x)
        return (v2.ToPILImage()(xtp / xtp.max())) if xtp is not None else self.st(x)
    
class FFDPhase(nn.Module):
    """Fake Facial Detection classifier phase of 2-phase model architecture, outputs ``1.0`` if prediction is real, ``0.0`` if fake."""
    def __init__(self, d_input:int=32, d_output:int=64):
        super(FFDPhase, self).__init__()
        self.d_input, self.d_output = d_input, d_output
        self.conv1 = nn.Conv2d(3, d_input, kernel_size=(3,3), stride=(1,1))
        self.conv2 = nn.Conv2d(d_input, 128, kernel_size=(3,3), stride=(1,1))
        self.fc1 = nn.Linear(128*38*38, d_output)
        self.fc2 = nn.Linear(d_output, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
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

def model_feasibility_test(model:nn.Module, input_size:tuple[int, int, int]=(3,160,160)) -> bool:
    """Generates a tensor of ``input_size`` and runs it through a single forward pass of the provided ``model`` to confirm matrix multiplication structure is coherent, returning ``True`` if successful, ``False`` otherwise."""
    input_tensor = torch.randn(1, *input_size)
    try:
        output_tensor = model(input_tensor)
        print(f"Success: {output_tensor}")
        return True
    except Exception as e:
        print(f"Failed due to: '{e}'")
        return False
        
if __name__ == '__main__':
    # ffd = FakeFaceDetectorDevelopment(d_input=48, d_output=64) # 1625056
    ffd = FakeFaceDetectorDevelopmentDebug(d_input=48, d_output=64) # 1625056
    model_feasibility_test(ffd, input_size=(3,160,160))
