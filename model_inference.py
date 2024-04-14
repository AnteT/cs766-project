import torch
import torch.nn as nn
from torchvision.transforms import v2
from model import FFDPhase
from model import FFXPhase
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
    
def run_inference(image_path:str, ffx:v2.Transform=FFXPhase, ffd:nn.Module=FFDPhase, ffd_path:str='results-best/T0.885-huge+leaky+decay+deepconv-ffd+fe-s50000-e10-lr1e-04-din48-dout64-sd3.pt', display_result:bool=False) -> tuple[int, float]:
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
    print(f"{result} ({prob:.4f})")
    if display_result:
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
        fig.suptitle(f"Result of FFD phase on FFX: {result} ({(1-prob if result == "Fake" else prob):.4f})\n", fontsize=16)
        plt.tight_layout()
        plt.show()    
    return pred.item(), prob

def run_bulk_inference():
    test_images = [
        'dataset/140k/real_vs_fake/real-vs-fake/test/fake/NVMFA7F52H.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/fake/6H686M730J.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/fake/2CBCMJ93TF.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/fake/4H13NVCXRU.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/fake/ROUSNVF8NZ.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/fake/7GMRJ8XY1E.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/fake/G4185199D1.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/fake/AV7F0USBRY.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/fake/VP849XISYO.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/fake/3986S7FE8O.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/real/18233.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/real/54317.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/real/40155.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/real/12875.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/real/52543.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/real/11982.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/real/38779.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/real/28723.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/real/33489.jpg'
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/real/55753.jpg'
    ]
    num_correct, num_total, cum_offset = 0, 0, 0.0
    for i,image in enumerate(test_images):
        num_total += 1
        res, prob = run_inference(image_path=image, display_result=False)
        if res == 0 and i < 10:
            num_correct += 1
            cum_offset += (prob)
        elif res == 1 and i >= 10:
            num_correct += 1
            cum_offset += (1 - prob)
    print(f'Results: {(num_correct/num_total):.2%} ({num_correct} of {num_total}) with cumulative offset: {cum_offset:.4f}')    

if __name__ == '__main__':
    run_bulk_inference()
