import torch
import torch.nn as nn
from torchvision.transforms import v2
from model import FFXPhase
from model import FakeFaceDetectorDevelopment
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def run_inference(image_path:str, ffx:v2.Transform=FFXPhase,ffd:nn.Module=FakeFaceDetectorDevelopment, ffd_path:str='results-best/T0.885-huge+leaky+decay+deepconv-ffd+fe-s50000-e10-lr1e-04-din48-dout64-sd3.pt'):
    """Run inference using the 2-phase FFX+FFD model on the provided image path"""
    image = Image.open(image_path)
    ffx = ffx(fail_thresholds=[0.6, 0.7, 0.7])
    ffd = ffd(d_input=48, d_output=64)
    ffd.load_state_dict(torch.load(ffd_path))
    ffd.eval()   
    face = ffx(image)
    print(face, type(face))
    # face = face.unsqueeze(0) if len(face.shape) ==3 else face
    output = ffd(face.unsqueeze(0) if len(face.shape) ==3 else face).item()
    pred = "Real" if output else "Fake"
    print(output)
    print(pred)
    face = np.transpose(np.array(face),(1,2,0))
    face = face - face.min()
    face = face / face.max()
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)
    axes[0].set_title('Input image:')
    axes[0].axis('off')
    axes[1].imshow(face)
    axes[1].set_title('FFX phase:')
    axes[1].axis('off')
    fig.suptitle(f"Result of FFD phase on FFX: {pred}\n", fontsize=16)
    plt.tight_layout()
    plt.show()    

if __name__ == '__main__':
    # model_path = 'results-best/T0.885-huge+leaky+decay+deepconv-ffd+fe-s50000-e10-lr1e-04-din48-dout64-sd3.pt' # Best
    test_image = 'kiki-and-me-original.png'
    run_inference(image_path=test_image)