import torch
import torch.nn as nn
from torchvision.transforms import v2
from model import FFDPhase
from model import FFXPhase
from model import FakeFaceDetectorFE as FFDOnly
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
    
def run_inference(image_path:str, use_ffx:bool=True, ffx:v2.Transform=FFXPhase, ffd:nn.Module=FFDPhase, ffd_path:str='results-best/T0.885-huge+leaky+decay+deepconv-ffd+fe-s50000-e10-lr1e-04-din48-dout64-sd3.pt', display_result:bool=False) -> tuple[int, float]:
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
    if use_ffx:
        ffx = ffx()
        face = ffx(image)
    else:
        face = image
    transform = v2.Compose([v2.ToImage()
                            ,v2.ToDtype(torch.uint8, scale=True)
                            ,v2.Resize(size=(160,160), antialias=True)
                            ,v2.ToDtype(torch.float32, scale=True)
                            ,v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ffd = ffd(d_input=48, d_output=64) if use_ffx else ffd()
    ffd.load_state_dict(torch.load(ffd_path))
    ffd.eval()   
    face = transform(face)
    output = ffd(face.unsqueeze(0) if len(face.shape) == 3 else face)
    prob = output.item()
    pred = (output.data > 0.5).float()
    result = "Real" if pred else "Fake"
    # print(f"{result} ({prob:.4f})")
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

if __name__ == '__main__':
    baseline = [
         'dataset/140k/real_vs_fake/real-vs-fake/test/fake/G4185199D1.jpg' # both correctly ID as Fake
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/fake/AV7F0USBRY.jpg' # both correctly ID as Fake
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/real/18233.jpg'      # both correctly ID as Real
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/real/54317.jpg'      # both correctly ID as Real
    ]
    fake_images = [
        'dataset/processed/test/fake/0BMMR954WG.jpg' # example of FFD+FFX, correctly classifying as fake while FFD miscorrectly identifies as Real
        ,'dataset/140k/real_vs_fake/real-vs-fake/test/fake/PG7VKQ4L9N.jpg' # example of FFD+FFX correctly classifying fake from unprocessed dataset used to train FFD only
        ,'dataset/real_and_fake_face/fake/easy_19_0111.jpg' # example 1 of FFD+FFX correctly classifying fake from foreign dataset (Photoshop)
        ,'dataset/real_and_fake_face/fake/easy_17_0011.jpg' # example 2 of FFD+FFX correctly classifying fake from foreign dataset (Photoshop)
    ]

    ffd_ffx_model_path = 'results-best/T0.885-huge+leaky+decay+deepconv-ffd+fe-s50000-e10-lr1e-04-din48-dout64-sd3.pt'
    ffd_model_path = 'results-best/T0.921-huge+leaky-ffd-s50000-e10-lr1e-04.pt'
    
    # for image in fake_images: # FFX+FFD versus FFD
    for image in baseline: # Baseline
        print(f"{f' {image} ':-^120}")
        pred_ffx_ffd, prob_ffx_ffd = run_inference(image, display_result=False, use_ffx=True, ffd=FFDPhase, ffd_path=ffd_ffx_model_path)
        pred_ffx_ffd = "Real" if pred_ffx_ffd else "Fake"
        pred_ffd, prob_ffd = run_inference(image, display_result=False, use_ffx=False, ffd=FFDOnly, ffd_path=ffd_model_path)
        pred_ffd = "Real" if pred_ffd else "Fake"
        print(f"FFX+FFD:   {pred_ffx_ffd}  ({prob_ffx_ffd:.4f})")
        print(f"FFD Only:  {pred_ffd}  ({prob_ffd:.4f})")

