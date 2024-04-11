from typing import Literal
import sys, time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
from model import FakeFaceDetectorFE
from SQLDataModel import SQLDataModel

DATASET_DIR = 'dataset/140k/real_vs_fake/real-vs-fake/'
PROCESSED_DIR = 'dataset/processed/'

def torch_tensor_to_numpy(torch_tensor:torch.Tensor) -> np.ndarray:
    """Converts the input `torch_tensor` into a `numpy.ndarray` using an image transformation to transpose `(Channels, Height, Width)` into `(Height, Width, Channels)`."""
    if (arg_ndim := torch_tensor.ndim) != 3:
        raise ValueError(
            f"Invalid dimensions '{arg_ndim}', value for `torch_tensor` must have 3 dimensions representing `(C, H, W)` to be able to convert into numpy array"
            )
    arr = torch_tensor.numpy().transpose((1,2,0))
    # arr = (arr * 255).astype(np.uint8)
    return arr

def inspect_random_image(display_image:bool=False, use_feature_extract:bool=False, normalize:bool=False, seed:int=42) -> None:
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
    torch.manual_seed(seed)
    loader, _, _ = get_image_dataloaders(1, use_feature_extract=use_feature_extract,batch_size=1,normalize_tensors=normalize)
    for image, label in loader:
        image, label = image[0], label[0]
        image = torch_tensor_to_numpy(image)
        shape, dtype = image.shape, image.dtype
        min_r, min_g, min_b = np.around(np.amin(image, axis=(0, 1)),2)
        avg_r, avg_g, avg_b = np.around(np.mean(image, axis=(0, 1)),2)
        max_r, max_g, max_b = np.around(np.amax(image, axis=(0, 1)),2)
        print(f"image {shape = } {dtype}")
        print(f"{min_r = }, {avg_r = }, {max_r = }")
        print(f"{min_g = }, {avg_g = }, {max_g = }")
        print(f"{min_b = }, {avg_b = }, {max_b = }")
        print(f"{label = } ({'real' if label else 'fake'})")
        if display_image:
            image = (image * 255).astype(np.uint8)
            cdtype = image.dtype
            print(f'image cast {dtype} -> {cdtype}')
            min_r, min_g, min_b = np.around(np.amin(image, axis=(0, 1)),2)
            avg_r, avg_g, avg_b = np.around(np.mean(image, axis=(0, 1)),2)
            max_r, max_g, max_b = np.around(np.amax(image, axis=(0, 1)),2)
            print(f"{min_r = }, {avg_r = }, {max_r = }")
            print(f"{min_g = }, {avg_g = }, {max_g = }")
            print(f"{min_b = }, {avg_b = }, {max_b = }")            
            plt.title(f"label of image: {label} ({'real' if label else 'fake'})")
            plt.imshow(image)
            plt.show()
        return

def get_image_dataloaders(max_samples:int=20_000, use_feature_extract:bool=False, normalize_tensors:bool=False, batch_size:int=32, images_resize:tuple=(160,160), images_mean:float=0.5, images_std:float=0.5, val_subset:float=0.2) -> tuple[DataLoader]:
    """Returns `140k/` dataset in the format: ``(train_loader, validation_loader, test_loader)``"""
    dataset_dir = DATASET_DIR if not use_feature_extract else PROCESSED_DIR

    train_dir = f"{dataset_dir}/train"
    valid_dir = f"{dataset_dir}/valid"
    test_dir = f"{dataset_dir}/test"
    
    if normalize_tensors:
        # transform = transforms.Compose([transforms.Resize(images_resize),transforms.ToTensor(),transforms.Normalize((images_mean,images_mean,images_mean),(images_std,images_std,images_std))])
        transform = v2.Compose([v2.ToImage()
                                ,v2.ToDtype(torch.uint8, scale=True)
                                ,v2.Resize(size=images_resize, antialias=True)
                                ,v2.ToDtype(torch.float32, scale=True)
                                ,v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # transform = transforms.Compose([transforms.Resize(images_resize),transforms.ToTensor()])
        transform = v2.Compose([v2.Resize(images_resize),v2.ToTensor()])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)
    train_subset = Subset(train_dataset, indices=torch.randperm(len(train_dataset))[:min(max_samples, len(train_dataset))])
    test_subset = Subset(test_dataset, indices=torch.randperm(len(test_dataset))[:min(max_samples, len(test_dataset))])
    valid_subset = Subset(valid_dataset, indices=torch.randperm(len(valid_dataset))[:min(round(max_samples*val_subset), len(valid_dataset))])
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
    return train_dataloader, valid_dataloader, test_dataloader

def run_training(max_samples:int=10_000, images_resize:tuple[int,int]=(160,160), num_epochs:int=10, lr:float=0.005, val_subset:float=0.2, training_label:str='dev', normalize_tensors:bool=False, use_feature_extract:bool=False, save_model:bool=True, save_epoch_report:bool=True, save_batch_report:bool=False) -> None:
    """Run the primary training loop with `max_samples` using `lr` specified for the `num_epochs` provided optionally saving model to `save_model` if provided."""
    report_batch = {'Epoch':'int', 'Batch':'int', 'Type':'str','Loss':'float'}
    sdm_batch = SQLDataModel(dtypes=report_batch)
    report_epoch = {'Epoch':'int', 'Train Loss':'float', 'Validation Loss':'float', 'Validation Accuracy':'float', 'Time (seconds)':'float'}
    sdm = SQLDataModel(dtypes=report_epoch)
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    train_loader, valid_loader, test_loader = get_image_dataloaders(max_samples=max_samples, normalize_tensors=normalize_tensors, use_feature_extract=use_feature_extract, batch_size=32, images_resize=images_resize, val_subset=val_subset)
    print(f"cuda device available: {has_cuda}")
    
    model = FakeFaceDetectorFE().to(device=device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_batch_idx, val_batch_idx = 0, 0
    for epoch in range(num_epochs):
        start_time = time.time()  # Start timer for epoch        
        running_loss = 0.0
        
        # Training phase
        model.train()
        for (inputs, labels) in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)            
            labels = labels.view(-1, 1) # reshape for functional sigmoid activation     
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_batch_loss = loss.item()
            train_batch_idx += 1
            sdm_batch[sdm_batch.row_count] = [epoch+1,train_batch_idx,'training',round(train_batch_loss,6)]
        epoch_time = time.time() - start_time    
        training_loss = (running_loss/len(train_loader))         

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for (inputs, labels) in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
                val_batch_loss = loss.item()
                val_batch_idx += 1
                sdm_batch[sdm_batch.row_count] = [epoch+1,val_batch_idx,'validation',round(val_batch_loss,6)]
                predicted = (outputs.data > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        validation_loss = (val_loss/len(valid_loader))
        validation_accuracy = correct / total
        print(f'Epoch {epoch+1}, Training Loss: {training_loss:.4f}, Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_accuracy:.2%} ({correct} of {total}), Time: {epoch_time:.4f}')
        epoch_data = [epoch+1, round(training_loss,6), round(validation_loss,6), round(validation_accuracy,6), round(epoch_time,6)]
        sdm[epoch] = epoch_data
    print(sdm)
    if save_batch_report:
        print(sdm_batch)
    report_summary = f"""(samples: {max_samples}, epochs: {num_epochs}, learing rate: {lr:.0e}, normalized tensors: {normalize_tensors}, used feature extraction: {use_feature_extract})"""
    report_label = f"{training_label}-s{max_samples}-e{num_epochs}-lr{lr:.0e}.csv" if not use_feature_extract else f"s{max_samples}-e{num_epochs}-lr{lr:.0e}+fe.csv"
    if save_epoch_report:
        sdm.to_csv(f"results/epoch-report-{report_label}")
    if save_batch_report:
        sdm_batch.to_csv(f"results/batch-report-{report_label}")
    # Test the model
    model.eval()  # set the model to evaluation mode
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
        report_summary = f'Test Accuracy: {correct / total:.2%} ({correct} of {total}), Test Loss: {(test_loss/len(test_loader)):.4f} {report_summary}'
    if save_model:
        model_label = f"{training_label}-ffd{'+fe' if use_feature_extract else ''}-s{max_samples}-e{num_epochs}-lr{lr:.0e}.pt"
        torch.save(model.state_dict(), model_label) 
        print(f"model saved to '{model_label}'")
    model.train()
    print(f'Finished with summary:\n{report_summary}')


if __name__ == '__main__':
    # NOTE: Best result:
    # Test Accuracy: 83.55% (8355 of 10000), Test Loss: 0.4551 (samples: 10000, epochs: 10, learing rate: 1e-04, normalized tensors: True, used feature extraction: False)
    FEATURE_EXTRACT = True
    NORMALIZE_TENSORS = True
    run_training(training_label='compare+leaky',max_samples=10_000,use_feature_extract=FEATURE_EXTRACT,normalize_tensors=NORMALIZE_TENSORS,images_resize=(160,160),num_epochs=10,lr=1e-4,save_model=True,save_epoch_report=True) 
    
    # inspect_random_image(display_image=True, use_feature_extract=False, normalize=True, seed=4)
