import os, random, glob
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

DATASET_DIR = 'dataset/140k/real_vs_fake/real-vs-fake/'
dataset_dir = DATASET_DIR
train_dir = f"{dataset_dir}/train"
test_dir = f"{dataset_dir}/test"
valid_dir = f"{dataset_dir}/valid"

def get_mtcnn_model(image_size:int=160) -> MTCNN:
    """Returns the instantiated MTCNN model for multitask convolutional neural network facial detection and resizes image to provided `image_size`."""
    mtcnn = MTCNN(image_size=image_size, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False)
    return mtcnn

def preprocess_image(image_path:str, output_dir:str, save_as:str=None, display_image:bool=False):
    image = Image.open(image_path)
    mtcnn = get_mtcnn_model()
    face = mtcnn(image)
    if face is not None:
        face = face / face.max()
        face = ToPILImage()(face)
        if display_image:
            plt.imshow(face)
            plt.show()
        if save_as is None:
            output_path = os.path.join(output_dir, os.path.basename(image_path))
        else:
            output_path = save_as
        face.save(output_path)

def preprocess_image_directory_and_save_output(input_dir:str, output_dir:str, subset:int=10_000) -> None:
    """Preprocesses a randomly selected `subset` of images from `input_dir` and saves result to `output_dir`."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_paths = glob.glob(os.path.join(input_dir, '*'))
    image_paths = random.sample(image_paths, min(len(image_paths), subset))
    for i, image_path in enumerate(image_paths):
        img_label = f"{output_dir}/{image_path.split('\\')[-1]}"
        print(f"\r{i+1:03}/{subset} | {img_label:<46.46} | {((i+1)/subset*100):.2f}%", end='')
        preprocess_image(image_path, output_dir)        

def bulk_process_directories(dirs:tuple[str, str, int]) -> None:
    """
    Bulk process all images in `dir` according to the format `(from_dir, to_dir, num_images)`. 
    
    Example::

        # Expected input structure for dirs argument:
        target_dirs = [
            (f"{train_dir}/fake",'dataset/processed/train/fake', 50_000)
            ,(f"{train_dir}/real",'dataset/processed/train/real', 50_000)
            ,(f"{test_dir}/fake",'dataset/processed/test/fake', 5_000)
            ,(f"{test_dir}/real",'dataset/processed/test/real', 5_000)
            ,(f"{valid_dir}/fake",'dataset/processed/valid/fake', 5_000)
            ,(f"{valid_dir}/real",'dataset/processed/valid/real', 5_000)
        ]

        # Process all images in specified directories up to target_dirs[-1] images:
        bulk_process_directories(target_dirs)
    """
    for input_dir, output_dir, limit in dirs:
        batch_label = output_dir.split('processed')[-1]
        print(f'Running {batch_label}:')
        preprocess_image_directory_and_save_output(input_dir, output_dir, subset=limit)
        print(f'\nFinished.')

if __name__ == '__main__':
    fe_input = 'C:/Users/212765834/Desktop/Personal Repos/CS766 - Computer Vision/CS766 - CV Final Project/kiki-and-me-original.png'
    fe_output = 'C:/Users/212765834/Desktop/Personal Repos/CS766 - Computer Vision/CS766 - CV Final Project/presentation-docs'
    fe_name = 'C:/Users/212765834/Desktop/Personal Repos/CS766 - Computer Vision/CS766 - CV Final Project/kiki-and-me-result.png'
    preprocess_image(fe_input, fe_output, save_as=fe_name, display_image=True)
