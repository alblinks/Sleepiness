import os
import random
import shutil


if __name__ == "__main__":

    # Define the source directory containing subfolders of images
    source_directory = "/home/mwaltz/train"

    # Define the destination directory to save sampled images
    destination_directory = "/home/mwaltz/sampleImages/raw"

    # Define the number of images to sample
    num_images_to_sample = 50_000

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Initialize a list to store sampled image paths
    sampled_image_paths = []

    # Iterate over all subdirectories and collect image paths
    for root, dirs, files in os.walk(source_directory):
        for file in files:
            # Check if the file is an image (you may want to refine this check based on your image file extensions)
            if file.endswith('.jpg'):
                # Construct the path to the current image
                image_path = os.path.join(root, file)
                # Add the image path to the list
                sampled_image_paths.append(image_path)

    # Shuffle the list of image paths
    random.shuffle(sampled_image_paths)

    # Sample exactly num_images_to_sample images
    for i, image_path in enumerate(sampled_image_paths[:num_images_to_sample], start=1):

        if i % 1000 == 0:
            print(f"{i} images sampled.")

        # Construct the new filename with prefix
        _, filename = os.path.split(image_path)
        new_filename = f"{i}_{filename}"
        # Construct the destination path
        destination_path = os.path.join(destination_directory, new_filename)
        # Copy the sampled image to the destination directory with the new filename
        shutil.copy(image_path, destination_path)
