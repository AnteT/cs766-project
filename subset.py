import os
import random
import shutil

def create_subset_dataset(root_dir, output_dir, subset_size=10000):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_dir_real = os.path.join(output_dir, 'real')
    output_dir_fake = os.path.join(output_dir, 'fake')
    if not os.path.exists(output_dir_real):
        os.makedirs(output_dir_real)
    if not os.path.exists(output_dir_fake):
        os.makedirs(output_dir_fake)

    # Define the subdirectories for real and fake images
    real_dir = os.path.join(root_dir, 'real')
    fake_dir = os.path.join(root_dir, 'fake')

    # Get the list of files in the real and fake directories
    real_files = os.listdir(real_dir)
    fake_files = os.listdir(fake_dir)

    # Randomly select subset_size number of images from each directory
    selected_real_files = random.sample(real_files, min(subset_size, len(real_files)))
    selected_fake_files = random.sample(fake_files, min(subset_size, len(fake_files)))

    # Copy selected files to the output directory
    for file in selected_real_files:
        src = os.path.join(real_dir, file)
        dst = os.path.join(output_dir_real, file)
        shutil.copy(src, dst)

    for file in selected_fake_files:
        src = os.path.join(fake_dir, file)
        dst = os.path.join(output_dir_fake, file)
        shutil.copy(src, dst)

if __name__ == "__main__":
    root_dir = "./140k_real_vs_fake/real-vs-fake/train"
    output_dir = "./10krvf"
    subset_size = 10000
    create_subset_dataset(root_dir, output_dir, subset_size)
