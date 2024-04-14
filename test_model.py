import sys, time
import torch
from torch import nn, optim
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from model import FakeFaceDetector, FakeFaceDetectorDevelopment

DATASET_DIR = 'dataset/140k/real_vs_fake/real-vs-fake/'
PROCESSED_DIR = 'dataset/processed/'

def torch_tensor_to_numpy(torch_tensor:torch.Tensor) -> np.ndarray:
    """Converts the input `torch_tensor` into a `numpy.ndarray` using an image transformation to transpose `(Channels, Height, Width)` into `(Height, Width, Channels)`."""
    if (arg_ndim := torch_tensor.ndim) != 3:
        raise ValueError(
            f"Invalid dimensions '{arg_ndim}', value for `torch_tensor` must have 3 dimensions representing `(C, H, W)` to be able to convert into numpy array"
            )
    return torch_tensor.numpy().transpose((1,2,0))

def get_image_dataloaders(max_samples:int=20_000, use_feature_extract:bool=True, normalize_tensors:bool=True, batch_size:int=32, images_resize:tuple=(160,160), val_subset:float=0.2) -> tuple[DataLoader]:
    """Returns `140k/` dataset in the format: ``(train_loader, validation_loader, test_loader)``"""
    dataset_dir = DATASET_DIR if not use_feature_extract else PROCESSED_DIR
    test_dir = f"{dataset_dir}/test"
    if normalize_tensors:
        transform = v2.Compose([v2.ToImage()
                                ,v2.ToDtype(torch.uint8, scale=True)
                                ,v2.Resize(size=images_resize, antialias=True)
                                ,v2.ToDtype(torch.float32, scale=True)
                                ,v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = v2.Compose([v2.Resize(images_resize),v2.ToTensor()])
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_subset = Subset(test_dataset, indices=torch.randperm(len(test_dataset))[:min(max_samples, len(test_dataset))])
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return None, None, test_dataloader

def run_testing(model_path:str, model:nn.Module, max_samples:int=10_000, images_resize:tuple[int,int]=(64,64), seed:int=42) -> None:
    """Run the primary training loop with `max_samples` using `lr` specified for the `num_epochs` provided optionally saving model to `save_model` if provided."""
    torch.manual_seed(seed=seed)
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    _, _, test_loader = get_image_dataloaders(max_samples=max_samples, batch_size=32, images_resize=images_resize)
    model = model.to(device=device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    criterion = nn.BCELoss()
    # Test the model
    test_loss, correct, total = 0.0, 0, 0
    TP, TN, FP, FN = 0, 0, 0, 0
    zf = 12
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            test_loss += loss.item()
            predicted = (outputs.data > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            TP += ((predicted == 1) & (labels == 1)).sum().item()
            TN += ((predicted == 0) & (labels == 0)).sum().item()
            FP += ((predicted == 1) & (labels == 0)).sum().item()
            FN += ((predicted == 0) & (labels == 1)).sum().item()            
        report_summary = f'''{'Results:':<{zf}} {correct} of {total}\n{f'Accuracy:':<{zf}} {correct / total:.2%}\n{f'Loss:':<{zf}} {(test_loss/len(test_loader)):.4f}'''
    model.train()
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    report_summary = f'''{report_summary}\n{f'TP:':<{zf}} {TP}\n{f'FP:':<{zf}} {FP}\n{f'FN:':<{zf}} {FN}\n{f'TN:':<{zf}} {TN}\n{f'Precision:':<{zf}} {precision:.2f}\n{f'Recall:':<{zf}} {recall:.2f}\n{f'F1-Score:':<{zf}} {f1_score:.2f}'''
    print(f"{' Test Summary ':-^120}")
    print(f'''{f'Model:':<{zf}} {model_path}\n{report_summary}''')
    print('-'*120)

if __name__ == '__main__':
    # saved_models = [
    #     'models/ffd-s5000-e20-lr3e-04.pt'
    #     ,'models/ffd-s5000-e50-lr3e-04.pt'
    #     ,'models/ffd-s5000-e80-lr1e-04.pt'
    #     ,'models/ffd-s5000-e80-lr2e-04.pt'
    #     ,'models/ffd-s10000-e50-lr3e-04.pt'
    # ]
    # for model in saved_models:
    #     run_testing(model, max_samples=1000)

    # model_path = 'results-best/T0.882-huge+leaky+decay+deepconv-ffd+fe-s50000-e10-lr1e-04-din32-dout64-sd3.pt' # Second best
    model_path = 'results-best/T0.885-huge+leaky+decay+deepconv-ffd+fe-s50000-e10-lr1e-04-din48-dout64-sd3.pt' # Best
    model = FakeFaceDetectorDevelopment(d_input=48, d_output=64)
    run_testing(model_path=model_path, model=model, max_samples=50_000, images_resize=(160,160), seed=35)