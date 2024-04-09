import os, random, glob
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision.transforms import ToPILImage

DATASET_DIR = 'dataset/140k/real_vs_fake/real-vs-fake/'
dataset_dir = DATASET_DIR
train_dir = f"{dataset_dir}/train"
test_dir = f"{dataset_dir}/test"
valid_dir = f"{dataset_dir}/valid"


mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False)

def preprocess_image(image_path, mtcnn, OUTPUT_DIR):
    image = Image.open(image_path)
    face = mtcnn(image)
    face = face / face.max()
    if face is not None:
        face = ToPILImage()(face)
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(image_path))
        face.save(output_path)

def preprocess_image_directory_and_save_output(input_dir:str, output_dir:str, subset:int=10_000) -> None:
    """Preprocesses a randomly selected `subset` of images from `input_dir` and saves result to `output_dir`."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_paths = glob.glob(os.path.join(input_dir, '*'))
    sampled_image_paths = random.sample(image_paths, min(len(image_paths), subset))
    for i, image_path in enumerate(image_paths[:subset]):
        img_label = f"{output_dir}/{image_path.split('\\')[-1]}"
        # print(f"\r{i+1:03}/{subset} | {image_path:>66.66}...", end='')
        print(f"\r{i+1:03}/{subset} | {img_label:<40.40} | {((i+1)/subset*100):.2f}%", end='')
        preprocess_image(image_path, mtcnn, output_dir)        

remaining_dirs = [
    (f"{train_dir}/fake",'dataset/processed/train/fake', 10_000)
    ,(f"{train_dir}/real",'dataset/processed/train/real', 10_000)
    ,(f"{test_dir}/fake",'dataset/processed/test/fake', 5_000)
    ,(f"{test_dir}/real",'dataset/processed/test/real', 5_000)
    ,(f"{valid_dir}/fake",'dataset/processed/valid/fake', 5_000)
    ,(f"{valid_dir}/real",'dataset/processed/valid/real', 5_000)
]

for INPUT_DIR, OUTPUT_DIR, ARB_LIMIT in remaining_dirs:
    preprocess_image_directory_and_save_output(INPUT_DIR, OUTPUT_DIR, subset=5)

#     if not os.path.exists(OUTPUT_DIR):
#         os.makedirs(OUTPUT_DIR)
#     image_paths = glob.glob(os.path.join(INPUT_DIR, '*'))
#     sampled_image_paths = random.sample(image_paths, min(len(image_paths), ARB_LIMIT))
#     for i, image_path in enumerate(image_paths[:ARB_LIMIT]):
#         print(f"\r{i+1:03}/{ARB_LIMIT} | {image_path:>66.66}...", end='')
#         preprocess_image(image_path, mtcnn, OUTPUT_DIR)        