import sys, time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from model import FakeFaceDetector
from SQLDataModel import SQLDataModel
from facenet_pytorch import MTCNN
from face_isolation.face_extraction_transform import FaceExtractionTransform

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
    
    # Original transform operations
    # transform = transforms.Compose([transforms.Resize(images_resize),transforms.ToTensor(),transforms.Normalize((images_mean,images_mean,images_mean),(images_std,images_std,images_std))])

    # Define the face extraction transform
    face_extraction_transform = FaceExtractionTransform(images_resize)

    transform = transforms.Compose([
        face_extraction_transform,
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

def run_training(max_samples:int=10_000, num_epochs:int=10, lr:float=0.005, save_model:bool=True) -> None:
    """Run the primary training loop with `max_samples` using `lr` specified for the `num_epochs` provided optionally saving model to `save_model` if provided."""
    report_batch = {'Epoch':'int', 'Batch':'int', 'Type':'str','Loss':'float'}
    sdm_batch = SQLDataModel(dtypes=report_batch)
    report_epoch = {'Epoch':'int', 'Train Loss':'float', 'Validation Loss':'float', 'Validation Accuracy':'float', 'Time (seconds)':'float'}
    sdm = SQLDataModel(dtypes=report_epoch)
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    train_loader, valid_loader, test_loader = get_image_dataloaders(max_samples=max_samples, batch_size=32, images_resize=(64,64))
    print(f"cuda device available: {has_cuda}")
    
    model = FakeFaceDetector().to(device=device)
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
    print('-'*120)
    print(f'Training Finished ({max_samples} samples, {num_epochs} epochs, learing rate {lr:.0e})')
    print('Batch report data:')
    print(sdm_batch)
    print('Epoch report data:')
    print(sdm)
    report_label = f"s{max_samples}-e{num_epochs}-lr{lr:.0e}.csv"
    sdm.to_csv(f"results/epoch-report-{report_label}")
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
        print(f'Test Loss: {(test_loss/len(test_loader)):.4f}, Test Accuracy: {correct / total:.2%} ({correct} of {total})')

    if save_model:
        model_label = f"ffd-s{max_samples}-e{num_epochs}-lr{lr:.0e}.pt"
        torch.save(model.state_dict(), model_label) 
        print(f"model saved to '{model_label}'")
    model.train()
    print(f'finished')

def display_random_image():
    """Get one random image to test facial extraction transform in dataloaders."""
    train_loader, valid_loader, test_loader = get_image_dataloaders(max_samples=10, batch_size=32, images_resize=(64,64))
    # Get a random batch
    for images, labels in train_loader:
        # Convert tensor to numpy array and transpose dimensions
        image = images[0].permute(1, 2, 0).numpy()
        # Display the image
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        break  # Break after displaying one random image

if __name__ == '__main__':
    # run_training(max_samples=5_000, num_epochs=80, lr=1e-4, save_model=True) # do 1e-4 next
    display_random_image()