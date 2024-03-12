import os
import random
import shutil

# Source directory containing the images
source_dir = "/home/mwaltz/balanced/eyes_aug_train/open_aug"

# Destination directory to move the sampled images
destination_dir = "/home/mwaltz/balanced/eyes_aug_test/open_aug"

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# List all the image files in the source directory
image_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Calculate the number of images to sample (20% of total)
num_images_to_sample = int(len(image_files) * 0.2)

# Sample 20% of the image files randomly
sampled_images = random.sample(image_files, num_images_to_sample)

# Move the sampled images to the destination directory
for image_file in sampled_images:
    source_path = os.path.join(source_dir, image_file)
    destination_path = os.path.join(destination_dir, image_file)
    shutil.move(source_path, destination_path)
    print(f"Moved {image_file} to {destination_dir}")
