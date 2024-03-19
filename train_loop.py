import sys, time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
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
    loader, _ = get_image_dataloaders(dataset_dir, batch_size=1)
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

def get_image_dataloaders(dirpath:str, tt_split:float=0.8, batch_size:int=32, images_resize:tuple=(128,128), images_mean:float=0.5, images_std:float=0.5) -> tuple[DataLoader]:
    """
    Get the images from the specified `dirpath` and return for training after applying transform using `images_mean`, `images_std` and `images_resize`. 
    
    Parameters:
        `dirpath` (str): The target directory for the image dataset
        `tt_split` (float): The train-test split to use for splitting the dataset, default is `0.8` for 80-20 split.
        `batch_size` (int): The batch size to use for loading data.
        `images_mean` (float): The mean to use for transform value, default is `0.442756` calculated from `get_image_mean_std()`.
        `images_std` (float): The std to use for transform value, default is `0.276322` calculated from `get_image_mean_std()`.
        `images_resize` (tuple): The resize to use for transform value, default is `(128, 128)` from estimate.
    
    Returns:
        `DataLoader` (tuple[DataLoader]): A tuple of dataloaders constructed using the transformation specified in the format of
            `(train_loader, test_loader)`            

    Note:
        Label of `1` signifies a real face, and `0` signifies a fake face.
    """
    transform = transforms.Compose([
        transforms.Resize(images_resize),
        transforms.ToTensor(),
        transforms.Normalize((images_mean,images_mean,images_mean),(images_std,images_std,images_std))
    ])    
    dataset = datasets.ImageFolder(root=dirpath, transform=transform)
    train_size = int(tt_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def run_training(num_epochs:int=10, lr:float=0.005) -> None:
    """Run the primary training loop using `lr` specified for the `num_epochs` provided."""
    from model import FakeFaceDetector

    dataset_dir = 'dataset/real_and_fake_face'
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    train_loader, test_loader = get_image_dataloaders(dataset_dir, tt_split=0.8, batch_size=32, images_resize=(64,64))

    # Check if cuda is found and available
    print(f"cuda device available: {has_cuda}")

    model = FakeFaceDetector().to(device=device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()  # Start timer for epoch        
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
        epoch_time = time.time() - start_time             
        print(f"Epoch {epoch+1}, Loss: {(running_loss/len(train_loader)):.6f}, Time: {epoch_time:.6f}")   

    # Test the model
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.view(-1, 1)
            outputs = model(images)
            predicted = (outputs.data > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Epoch {epoch+1}, Test Accuracy: {correct / total:.4%} ({correct} correct out of {total})')
    model.train()  # set the model back to training mode        

if __name__ == '__main__':
    # inspect_random_image(display_image=True)
    run_training(num_epochs=20, lr=0.008)


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