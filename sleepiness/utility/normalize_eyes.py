import os
import cv2
import numpy as np


def min_max_scaling(image):
    # Convert the image to float32
    image = image.astype(np.float32)

    # Scale the image using min-max scaling
    min_val = np.min(image)
    max_val = np.max(image)
    scaled_image = (image - min_val) / (max_val - min_val) * 255

    # Convert back to uint8 and return
    return scaled_image.astype(np.uint8)

# Function to convert RGB images to binary (black and white) and save them
def convert_to_binary(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop over the images in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Read the image
            image_path = os.path.join(input_dir, filename)
            img = cv2.imread(image_path)

            # Resize the image
            img = cv2.resize(img, (50, 20))

            # Convert the image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # MinMax scaling
            gray_img = min_max_scaling(gray_img)

            # Apply binary thresholding
            #_, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

            # Save the binary image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, gray_img)

# Input and output directories
input_dir = "/home/mwaltz/sampleImages/eyes_from_faces"
output_dir = "/home/mwaltz/sampleImages/eyes_from_faces_processed"

# Convert RGB images to binary (black and white) and save them
convert_to_binary(input_dir, output_dir)
