import os
import random
import shutil

def split_images(top_folder, train_folder, test_folder, split_ratio=0.8):
    # Create train and test folders if they don't exist
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Iterate over subfolders and images in the top folder
    for root, dirs, files in os.walk(top_folder):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            # Create train and test subfolders for the current subfolder
            train_subdir = os.path.join(train_folder, os.path.relpath(subdir_path, top_folder))
            test_subdir = os.path.join(test_folder, os.path.relpath(subdir_path, top_folder))
            os.makedirs(train_subdir, exist_ok=True)
            os.makedirs(test_subdir, exist_ok=True)
            
            # Collect image filenames in the current subfolder
            images = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
            
            # Randomly shuffle the list of image filenames
            random.shuffle(images)
            
            # Split the images into train and test sets
            split_index = int(len(images) * split_ratio)
            train_images = images[:split_index]
            test_images = images[split_index:]
            
            # Move train images to the train folder
            for img in train_images:
                src_path = os.path.join(subdir_path, img)
                dest_path = os.path.join(train_subdir, img)
                shutil.move(src_path, dest_path)
            
            # Move test images to the test folder
            for img in test_images:
                src_path = os.path.join(subdir_path, img)
                dest_path = os.path.join(test_subdir, img)
                shutil.move(src_path, dest_path)

# Example usage
top_folder = '/home/mwaltz/LGCS_Daten_Video_Teil1_gelabelt'  # Path to the top folder containing subject subfolders
train_folder = '/home/mwaltz/train'  # Path to the folder to save train images
test_folder = '/home/mwaltz/test'  # Path to the folder to save test images

# Call the function to split images into train and test sets
split_images(top_folder, train_folder, test_folder, split_ratio=0.8)
