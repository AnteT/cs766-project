from typing import Literal
import sys, time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
from model import FakeFaceDetectorFE, FakeFaceDetectorDevelopment
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

    Example::

        # This will display a randomly selected image
        inspect_random_image(display_image=True, use_feature_extract=False, normalize=True, seed=4)

    Note:
        - Image details and normalization may render images with discoloration, ensure proper type conversions and normalizations.
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

def get_image_dataloaders(max_samples:int=20_000, use_feature_extract:bool=False, normalize_tensors:bool=False, batch_size:int=32, images_resize:tuple=(160,160), val_subset:float=0.2) -> tuple[DataLoader]:
    """Returns `140k/` dataset in the format: ``(train_loader, validation_loader, test_loader)``"""
    dataset_dir = DATASET_DIR if not use_feature_extract else PROCESSED_DIR

    train_dir = f"{dataset_dir}/train"
    valid_dir = f"{dataset_dir}/valid"
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

def run_training(model:nn.Module, max_samples:int=10_000, images_resize:tuple[int,int]=(160,160), num_epochs:int=10, lr:float=0.0003, l2_decay:float=0.0001, val_subset:float=0.2, training_label:str='dev', normalize_tensors:bool=False, use_feature_extract:bool=False, save_model:bool=True, save_epoch_report:bool=True, save_batch_report:bool=False, seed:int=42) -> None:
    """
    Run the primary training loop for training artificially generated facial image classifier and optionally save model as .pth checkpoint.
    
    Parameters:
        ``model`` (nn.Module): The neural network model to be trained, see ``model.py`` for current classes.
        ``max_samples`` (int, optional): Maximum number of samples to use for training. Defaults to 10_000, max possible for dataset is 50_000.
        ``images_resize`` (tuple[int, int], optional): Resize dimensions for the input images. Defaults to (160, 160) which should be used for the current model structure.
        ``num_epochs`` (int, optional): Number of epochs for training. Defaults to 10.
        ``lr`` (float, optional): Learning rate for the optimizer. Defaults to 0.0003 or 1e-4.
        ``l2_decay`` (float, optional): L2 regularization weight decay. Defaults to 0.0001 or 1e-5.
        ``val_subset`` (float, optional): Percentage of training data to use for validation. Defaults to 0.2.
        ``training_label`` (str, optional): Label for the training phase. Defaults to 'dev'.
        ``normalize_tensors`` (bool, optional): Whether to normalize input tensors. Defaults to False.
        ``use_feature_extract`` (bool, optional): Whether to use feature extraction. Defaults to False.
        ``save_model`` (bool, optional): Whether to save the trained model. Defaults to True.
        ``save_epoch_report`` (bool, optional): Whether to save epoch-wise training reports. Defaults to True.
        ``save_batch_report`` (bool, optional): Whether to save batch-wise training reports. Defaults to False.
        ``seed`` (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        ``None``: Model state dict and CSV files may be created based on provided arguments.
    
    Note:
        - Value for ``max_samples`` determines the total number of images which will be used for both train and test loops.
        - Value for ``val_subset`` does not impact the size of the test dataset used for computing final result metrics.
        - Value for ``images_resize`` should not be changed without changing the corresponding convolutional layers present in the ``model`` parameter.
    """
    torch.manual_seed(seed=seed)
    report_batch = {'Epoch':'int', 'Batch':'int', 'Type':'str','Loss':'float'}
    sdm_batch = SQLDataModel(dtypes=report_batch, display_index=False)
    report_epoch = {'Epoch':'int', 'Train Loss':'float', 'Validation Loss':'float', 'Validation Accuracy':'float', 'Time (seconds)':'float'}
    sdm = SQLDataModel(dtypes=report_epoch, display_index=False, display_color='#A6D7E8')
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    train_loader, valid_loader, test_loader = get_image_dataloaders(max_samples=max_samples, normalize_tensors=normalize_tensors, use_feature_extract=use_feature_extract, batch_size=32, images_resize=images_resize, val_subset=val_subset)
    print(f'Started training with cuda = {has_cuda}...')
    model = model.to(device=device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_decay)
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
            if save_batch_report:
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
                if save_batch_report:
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
    report_summary = f"""(Samples: {max_samples}, Epochs: {num_epochs}, LR: {lr:.0e}, Normalized: {normalize_tensors}, FEX: {use_feature_extract}, Seed: {seed})"""
    report_label = f"{training_label}-s{max_samples}-e{num_epochs}-lr{lr:.0e}-din{model.d_input}-dout{model.d_output}-sd{seed}" if not use_feature_extract else f"+fe-s{max_samples}-e{num_epochs}-lr{lr:.0e}-din{model.d_input}-dout{model.d_output}-sd{seed}"
    # Test the model
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    TP, TN, FP, FN = 0, 0, 0, 0
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
        report_summary = f'Test Accuracy: {correct / total:.2%} ({correct} of {total}), Test Loss: {(test_loss/len(test_loader)):.4f} {report_summary}'
        report_accuracy_prefix = f"T{correct / total:.3f}"
    if save_epoch_report:
        sdm.to_csv(f"results/{report_accuracy_prefix}-epoch-report-{report_label}.csv")
    if save_batch_report:
        sdm_batch.to_csv(f"results/{report_accuracy_prefix}-batch-report-{report_label}.csv")
    if save_model:
        model_label = f"{report_accuracy_prefix}-{training_label}-ffd{'+fe' if use_feature_extract else ''}-s{max_samples}-e{num_epochs}-lr{lr:.0e}-din{model.d_input}-dout{model.d_output}-sd{seed}.pt"
        torch.save(model.state_dict(), model_label) 
        print(f"model saved to '{model_label}'")
    model.train()
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    report_summary = f'{report_summary}\nPrecision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f} ({TP = }, {FP = }, {FN = }, {TN = })'
    print(f'Finished with summary:\n{report_summary}')

if __name__ == '__main__':
    SEED = 3
    LABEL = 'tweakbest'
    LR = 1e-4
    L2_DECAY = 1e-5
    EPOCHS = 10
    MAX_SAMPLES = 50_000
    FEATURE_EXTRACT = True
    NORMALIZE_TENSORS = True
    SAVE_MODEL = True
    SAVE_REPORT = True
    SAVE_BATCH_REPORT = False
    
    # MODEL = FakeFaceDetectorDevelopment(d_input=48,d_output=48) # Shittier Test Accuracy: 86.89% (8687 of 9998)
    # MODEL = FakeFaceDetectorDevelopment(d_input=56,d_output=64) # Shittier Test Accuracy: 87.50% (8748 of 9998)
    # MODEL = FakeFaceDetectorDevelopment(d_input=48,d_output=96) # Shittier Test Accuracy: 87.99% (8797 of 9998)
    # MODEL = FakeFaceDetectorDevelopment(d_input=48,d_output=64) # Beeeesst Test Accuracy: 88.54% (8852 of 9998)
    # MODEL = FakeFaceDetectorDevelopment(d_input=48,d_output=56) # Shittier Test Accuracy: 86.10% (8608 of 9998)
    MODEL = FakeFaceDetectorDevelopment(d_input=48,d_output=72) # Run next

    run_training(model=MODEL,training_label=LABEL,max_samples=MAX_SAMPLES,use_feature_extract=FEATURE_EXTRACT,normalize_tensors=NORMALIZE_TENSORS,images_resize=(160,160),num_epochs=EPOCHS,lr=LR,l2_decay=L2_DECAY,save_model=SAVE_MODEL,save_epoch_report=SAVE_REPORT,val_subset=0.02, save_batch_report=SAVE_BATCH_REPORT, seed=SEED) 

"""
# NOTE: Best result :
┌───────┬────────────┬─────────────────┬─────────────────────┬────────────────┐
│ Epoch │ Train Loss │ Validation Loss │ Validation Accuracy │ Time (seconds) │
├───────┼────────────┼─────────────────┼─────────────────────┼────────────────┤
│     1 │     0.5417 │          0.4753 │              0.7620 │       455.1873 │
│     2 │     0.4151 │          0.3909 │              0.8220 │       415.8705 │
│     3 │     0.3223 │          0.3812 │              0.8250 │       452.0476 │
│     4 │     0.2471 │          0.3493 │              0.8510 │       420.9366 │
│     5 │     0.1848 │          0.3225 │              0.8560 │       459.7513 │
│     6 │     0.1321 │          0.3318 │              0.8600 │       437.3831 │
│     7 │     0.0933 │          0.3846 │              0.8570 │       450.0939 │
│     8 │     0.0631 │          0.3998 │              0.8650 │       445.3541 │
│     9 │     0.0421 │          0.4294 │              0.8610 │       507.4614 │
│    10 │     0.0330 │          0.4165 │              0.8740 │       466.5454 │
└───────┴────────────┴─────────────────┴─────────────────────┴────────────────┘
[10 rows x 5 columns]

Model: 'T0.885-huge+leaky+decay+deepconv-ffd+fe-s50000-e10-lr1e-04-din48-dout64-sd3.pt'
Summary: Test Accuracy: 88.54% (8852 of 9998), Test Loss: 0.4022 (Samples: 50000, Epochs: 10, LR: 1e-04, Normalized: True, FEX: True, Seed: 3)  
Conf Matrix: Precision: 0.86, Recall: 0.92, F1-Score: 0.89 (TP = 4604, FP = 751, FN = 395, TN = 4248)
"""