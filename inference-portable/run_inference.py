import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

def run_inference(image_path:str, ffx:v2.Transform=FFXPhase, ffd:nn.Module=FFDPhase, ffd_path:str='inference-portable/ffd.pt', display_result:bool=False, save_result:str=None) -> tuple[int, float]:
    """
    Run inference using the 2-phase FFX+FFD model on the provided image path and optionally displays the result.

    Parameters:
        ``image_path`` (str): The path to the image file to use for inference.
        ``ffx`` (v2.Transform): Phase 1 model for facial feature exraction.
        ``ffd`` (v2.Transform): Phase 2 model for fake facial detection.
        ``ffd_path`` (str): The path to the ``.pt`` file containing the state dict to use for evaluation.
        ``display_result`` (bool): If the original image, facial extraction, and result should be displayed.

    Returns:
        ``(prediction, prob)`` (tuple[int, float]): A tuple containing the prediction and the corresponding probability.

    Note:
        - Prediction ``1`` returned for real, ``0`` for fake.
    """
    image = Image.open(image_path)
    ffx = ffx()
    face = ffx(image)
    transform = v2.Compose([v2.ToImage()
                            ,v2.ToDtype(torch.uint8, scale=True)
                            ,v2.Resize(size=(160,160), antialias=True)
                            ,v2.ToDtype(torch.float32, scale=True)
                            ,v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ffd = ffd(d_input=48, d_output=64)
    ffd.load_state_dict(torch.load(ffd_path))
    ffd.eval()   
    face = transform(face)
    output = ffd(face.unsqueeze(0) if len(face.shape) == 3 else face)
    prob = output.item()
    pred = (output.data > 0.5).float()
    result = "Real" if pred else "Fake"
    conf = 1-prob if result == "Fake" else prob
    print(f"{image_path}: {result} ({conf:.4f})")
    if display_result:
        face = np.transpose(np.array(face),(1,2,0))
        face = face - face.min()
        face = face / face.max()
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[0].set_title('Input image:')
        axes[0].axis('off')
        axes[1].imshow(face)
        axes[1].set_title('FFX output:')
        axes[1].axis('off')
        fig.suptitle(f" Result of Inference with FFD phase on FFX: {result} ({conf:.4f}) ", fontsize=15)
        plt.tight_layout()
        if save_result is not None:
            plt.savefig(f'{save_result}',dpi=399)
        plt.show()    
    return pred.item(), prob    

def run_demo() -> None:
    """Run inference on the provided demonstration images."""
    fake_img = "./fake-face.jpg"
    run_inference(fake_img, display_result=True)
    real_img = "./real-face.jpg"
    run_inference(real_img, display_result=True)    

def main() -> None:
    """Primary entry-point for running inference through CLI arguments for one or more image paths."""
    parser = argparse.ArgumentParser(description='Run Facial Extraction & Fake Facial Detection inference on one or more images.')
    parser.add_argument('image_paths', nargs='+', help='Paths to the images to be processed, at least 1 is required.')
    parser.add_argument('-d', '--display', action='store_true', help='Display the images with the result.')
    args = parser.parse_args()
    image_paths = args.image_paths
    display = args.display
    for image_path in image_paths:
        run_inference(image_path=image_path, display_result=display)

if __name__ == '__main__':
    main()
    # python .\inference-portable\run_inference.py "inference-portable/fake-face.jpg"  "inference-portable/real-face.jpg"
    # fake_img = "inference-portable/fake-face.jpg"
    # run_inference(fake_img, display_result=True, save_result="C:/Users/212765834/Desktop/Personal Repos/CS766 - Computer Vision/CS766 - Webpage/cvapp/inference/inf-fake-face.png")
    # real_img = "inference-portable/real-face.jpg"
    # run_inference(real_img, display_result=True, save_result="C:/Users/212765834/Desktop/Personal Repos/CS766 - Computer Vision/CS766 - Webpage/cvapp/inference/inf-real-face.png")
    