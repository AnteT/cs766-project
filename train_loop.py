import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def torch_tensor_to_numpy(torch_tensor:torch.Tensor) -> np.ndarray:
    """Converts the input `torch_tensor` into a `numpy.ndarray` using an image transformation to transpose `(Channels, Height, Width)` into `(Height, Width, Channels)`."""
    if (arg_ndim := torch_tensor.ndim) != 3:
        raise ValueError(
            f"Invalid dimensions '{arg_ndim}', value for `torch_tensor` must have 3 dimensions representing `(C, H, W)` to be able to convert into numpy array"
            )
    return torch_tensor.numpy().transpose((1,2,0))

def inspect_random_image(display_image:bool=False) -> None:
    """
    Inspects a randomly selected `image` for debug information, printing the properties referenced below:

    Parameters:
        `display_image` (bool): If the randomly selected image should be displayed.
        
    Properties inspected:
        `shape`: from `image.shape` representing the shape of the input image.
        `dtype`: from `image.dtype` representing image value data type.
        `max_channel`: from `np.amax(image, axis=(0, 1))` as max values of each input channel.
            `max_r`: from `max_channel[0]` as red channel. 
            `max_g`: from `max_channel[1]` as green channel. 
            `max_b`: from `max_channel[2]` as blue channel. 
        `label`: from dataset representing real or fake image.

    Returns:
        `None`
    """
    dataset_dir = 'dataset/real_and_fake_face'
    loader = get_images(dataset_dir, batch_size=1)
    for image, label in loader:
        image, label = image[0], label[0]
        image = torch_tensor_to_numpy(image)
        shape, dtype = image.shape, image.dtype
        max_r, max_g, max_b = np.amax(image, axis=(0, 1))
        print(f"{shape = }")
        print(f"{dtype = }")
        print(f"{max_r = }")
        print(f"{max_g = }")
        print(f"{max_b = }")
        print(f"{label = } ({'real' if label else 'fake'})")
        if display_image:
            plt.title(f"label of image: {label} ({'real' if label else 'fake'})")
            plt.imshow(image)
            plt.show()
        return

def get_image_mean_std(dirpath:str) -> tuple[int]:
    """Gets the image dataset from `dirpath` and returns the pixel value mean and standard deviation as `(mean, std)`."""
    transform = transforms.Compose([transforms.ToTensor()])    
    data = datasets.ImageFolder(root=dirpath, transform=transform)    
    data_tensor = torch.concatenate([data[i][0] for i in range(len(data))])
    mean, std = torch.mean(data_tensor, dim=(0, 1)).mean(), torch.std(data_tensor, dim=(0, 1)).mean()
    return mean, std

def get_images(dirpath:str, batch_size:int=32, images_resize:tuple=(128,128), images_mean:float=0.5, images_std:float=0.5) -> DataLoader:
    """
    Get the images from the specified `dirpath` and return for training after applying transform using `images_mean`, `images_std` and `images_resize`. 
    
    Parameters:
        `dirpath` (str): The target directory for the image dataset
        `batch_size` (int): The batch size to use for loading data.
        `images_mean` (float): The mean to use for transform value, default is `0.442756` calculated from `get_image_mean_std()`.
        `images_std` (float): The std to use for transform value, default is `0.276322` calculated from `get_image_mean_std()`.
        `images_resize` (tuple): The resize to use for transform value, default is `(128, 128)` from estimate.
    
    Returns:
        `DataLoader`: A dataloader constructed using the transformation specified.

    Note:
        Label of `1` signifies a real face, and `0` signifies a fake face.
    """
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(images_resize),
        transforms.ToTensor(),
        transforms.Normalize((images_mean,images_mean,images_mean),(images_std,images_std,images_std))
    ])    
    data = datasets.ImageFolder(root=dirpath, transform=transform)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)    
    return loader

def run_training(num_epochs:int=10, lr:float=0.005) -> None:
    """Run the primary training loop using `lr` specified for the `num_epochs` provided."""
    from model import FakeFaceDetector

    dataset_dir = 'dataset/real_and_fake_face'
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    train_loader = get_images(dataset_dir, batch_size=32, images_resize=(64,64))

    # Check if cuda is found and available
    print(f"cuda device available: {has_cuda}")

    model = FakeFaceDetector().to(device=device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)            
            labels = labels.view(-1, 1) # reshape for functional sigmoid activation     
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")   

if __name__ == '__main__':
    # inspect_random_image(display_image=True)
    run_training(num_epochs=10, lr=0.5e-2)


"""
Output of first training loop at lr 0.005 for 10 epochs:

cuda device available: True
Epoch 1, Loss: 0.8924276977777481
Epoch 2, Loss: 0.6784730423241854
Epoch 3, Loss: 0.6633029114454985
Epoch 4, Loss: 0.655360559001565
Epoch 5, Loss: 0.6469162441790104
Epoch 6, Loss: 0.6355789313092828
Epoch 7, Loss: 0.624498943798244
Epoch 8, Loss: 0.6139185233041644
Epoch 9, Loss: 0.5922861993312836
Epoch 10, Loss: 0.570208353921771

"""