import sys, time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from model import FakeFaceDetector

DATASET_DIR = 'dataset/140k/real_vs_fake/real-vs-fake/'

def torch_tensor_to_numpy(torch_tensor:torch.Tensor) -> np.ndarray:
    """Converts the input `torch_tensor` into a `numpy.ndarray` using an image transformation to transpose `(Channels, Height, Width)` into `(Height, Width, Channels)`."""
    if (arg_ndim := torch_tensor.ndim) != 3:
        raise ValueError(
            f"Invalid dimensions '{arg_ndim}', value for `torch_tensor` must have 3 dimensions representing `(C, H, W)` to be able to convert into numpy array"
            )
    return torch_tensor.numpy().transpose((1,2,0))

def get_image_dataloaders(max_samples:int=20_000, batch_size:int=32, images_resize:tuple=(128,128), images_mean:float=0.5, images_std:float=0.5) -> tuple[DataLoader]:
    """Returns `140k/` dataset in the format: ``(train_loader, validation_loader, test_loader)``"""
    dataset_dir = DATASET_DIR
    train_dir = f"{dataset_dir}/train"
    test_dir = f"{dataset_dir}/test"
    valid_dir = f"{dataset_dir}/valid"
    transform = transforms.Compose([
        transforms.Resize(images_resize),
        transforms.ToTensor(),
        transforms.Normalize((images_mean,images_mean,images_mean),(images_std,images_std,images_std))
    ])
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)
    train_subset = Subset(train_dataset, indices=torch.randperm(len(train_dataset))[:min(max_samples, len(train_dataset))])
    test_subset = Subset(test_dataset, indices=torch.randperm(len(test_dataset))[:min(max_samples, len(test_dataset))])
    valid_subset = Subset(valid_dataset, indices=torch.randperm(len(valid_dataset))[:min(max_samples, len(valid_dataset))])
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
    return train_dataloader, valid_dataloader, test_dataloader

def run_testing(model_path:str, max_samples:int=10_000) -> None:
    """Run the primary training loop with `max_samples` using `lr` specified for the `num_epochs` provided optionally saving model to `save_model` if provided."""
    print(f'-'*100)
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    _, _, test_loader = get_image_dataloaders(max_samples=max_samples, batch_size=32, images_resize=(64,64))
    model = FakeFaceDetector().to(device=device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    criterion = nn.BCELoss()
    # Test the model
    test_loss, correct, total = 0.0, 0, 0
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
        print(f'`{model_path}` Loss: {(test_loss/len(test_loader)):.4f}, Accuracy: {correct / total:.2%} ({correct} of {total})')

if __name__ == '__main__':
    saved_models = [
        'models/ffd-s5000-e20-lr3e-04.pt'
        ,'models/ffd-s5000-e50-lr3e-04.pt'
        ,'models/ffd-s5000-e80-lr1e-04.pt'
        ,'models/ffd-s5000-e80-lr2e-04.pt'
        ,'models/ffd-s10000-e50-lr3e-04.pt'
    ]
    for model in saved_models:
        run_testing(model, max_samples=1000)