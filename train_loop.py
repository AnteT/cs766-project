import sys, time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset
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

def get_140k_image_dataloaders(total_size:int=20_000, tt_split:float=0.8, batch_size:int=32, images_resize:tuple=(128,128), images_mean:float=0.5, images_std:float=0.5) -> tuple[DataLoader]:
    DATASET_DIR = 'dataset/140k/real_vs_fake/real-vs-fake/train'
    transform = transforms.Compose([
        transforms.Resize(images_resize),
        transforms.ToTensor(),
        transforms.Normalize((images_mean,images_mean,images_mean),(images_std,images_std,images_std))
    ])   
    dataset = datasets.ImageFolder(root=DATASET_DIR, transform=transform)
    indices = torch.randperm(len(dataset))[:total_size]
    subset = Subset(dataset, indices)
    train_size = round(tt_split * len(subset))  # 80% for training
    test_size = len(subset) - train_size  # 20% for testing
    train_dataset, test_dataset = random_split(subset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

def run_training(num_epochs:int=10, lr:float=0.005, save_model:str=None) -> None:
    """Run the primary training loop using `lr` specified for the `num_epochs` provided optionally saving model to `save_model` if provided."""
    from model import FakeFaceDetector

    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    # dataset_dir = 'dataset/real_and_fake_face'
    # train_loader, test_loader = get_image_dataloaders(dataset_dir, tt_split=0.8, batch_size=32, images_resize=(64,64))
    train_loader, test_loader = get_140k_image_dataloaders(tt_split=0.8, batch_size=32, images_resize=(64,64))
    # Check if cuda is found and available
    print(f"cuda device available: {has_cuda} ({torch.version.cuda})")
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
    if save_model is not None:
        torch.save(model.state_dict(), save_model) 
        print(f"model saved to '{save_model}'")
    model.train()
    print(f'finished')

if __name__ == '__main__':
    # inspect_random_image(display_image=True)
    # run_training(num_epochs=50, lr=0.001, save_model='ffd-e50-lr1e3.pt') # 79.5% accuracy using lr: 1e-3, epochs: 50, total_size: 10_000, tt_split: 0.8, image_resize: (64,64)
    run_training(num_epochs=200, lr=0.0003, save_model='ffd-e200-lr3e4.pt') # run next total_size: 20_000


"""
Output of 140k dataset using CNN at lr 1e-3 for 50 epochs, first 12 epochs:

cuda device available: True (12.1)
Epoch 1, Loss: 0.676425, Time: 39.416640
Epoch 2, Loss: 0.590187, Time: 28.932080
Epoch 3, Loss: 0.529170, Time: 28.179731
Epoch 4, Loss: 0.484675, Time: 34.000679
Epoch 5, Loss: 0.436741, Time: 37.053384
Epoch 6, Loss: 0.395150, Time: 29.924166
Epoch 7, Loss: 0.355435, Time: 29.926557
Epoch 8, Loss: 0.310470, Time: 30.033522
Epoch 9, Loss: 0.278954, Time: 29.802948
Epoch 10, Loss: 0.235843, Time: 30.110958
Epoch 11, Loss: 0.195993, Time: 29.975606
Epoch 12, Loss: 0.159638, Time: 29.802055
Epoch 13, Loss: 0.139127, Time: 30.280042
Epoch 14, Loss: 0.101585, Time: 30.072499
Epoch 15, Loss: 0.084098, Time: 31.773986
Epoch 16, Loss: 0.074306, Time: 29.259795
Epoch 17, Loss: 0.048517, Time: 28.815021
Epoch 18, Loss: 0.040382, Time: 35.613732
Epoch 19, Loss: 0.074973, Time: 36.983704
Epoch 20, Loss: 0.042922, Time: 32.102381
Epoch 21, Loss: 0.024969, Time: 29.255951
Epoch 22, Loss: 0.042301, Time: 30.789441
Epoch 23, Loss: 0.030716, Time: 30.533591
Epoch 24, Loss: 0.064482, Time: 29.873001
Epoch 25, Loss: 0.011320, Time: 30.064110
Epoch 26, Loss: 0.007316, Time: 29.012978
Epoch 27, Loss: 0.002674, Time: 28.804330
Epoch 28, Loss: 0.001354, Time: 28.964646
Epoch 29, Loss: 0.000855, Time: 28.601851
Epoch 30, Loss: 0.000682, Time: 28.229314
Epoch 31, Loss: 0.000544, Time: 29.412738
Epoch 32, Loss: 0.000450, Time: 27.911226
Epoch 33, Loss: 0.000384, Time: 28.122136
Epoch 34, Loss: 0.000321, Time: 27.650081
Epoch 35, Loss: 0.000277, Time: 27.728242
Epoch 36, Loss: 0.000239, Time: 29.589487
Epoch 37, Loss: 0.000207, Time: 31.140086
Epoch 38, Loss: 0.000176, Time: 31.100951
Epoch 39, Loss: 0.000151, Time: 31.425896
Epoch 40, Loss: 0.000131, Time: 32.396354
Epoch 41, Loss: 0.000113, Time: 29.972846
Epoch 42, Loss: 0.000098, Time: 29.210037
Epoch 43, Loss: 0.000086, Time: 29.933231
Epoch 44, Loss: 0.000073, Time: 30.755196
Epoch 45, Loss: 0.000063, Time: 31.636939
Epoch 46, Loss: 0.000055, Time: 41.926067
Epoch 47, Loss: 0.000048, Time: 43.543257
Epoch 48, Loss: 0.000041, Time: 43.595753
Epoch 49, Loss: 0.000036, Time: 43.090540
Epoch 50, Loss: 0.000031, Time: 33.583533
Epoch 50, Test Accuracy: 79.5000% (1590 correct out of 2000)
model saved to 'ffd-e50-lr1e3.pt'

Output of first training loop at lr 0.005 for 10 epochs:

cuda device available: True (12.1)
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