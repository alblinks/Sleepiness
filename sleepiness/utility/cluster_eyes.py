import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# Function to load and preprocess images
def load_images(input_dir):
    images = []
    filenames = []  # To keep track of filenames
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            img = cv2.imread(image_path)
            images.append(img.flatten())     # Flatten image and add to list
            filenames.append(filename)       # Keep track of filenames
    return np.array(images), filenames

# Input directory containing 50x20 images
input_dir = "/home/mwaltz/sampleImages/eyes_processed"

# Load and preprocess images
images, filenames = load_images(input_dir)

# Define the number of clusters
num_clusters = 2

# Define the clustering pipeline
pipeline = make_pipeline(PCA(n_components=50), KMeans(n_clusters=num_clusters, random_state=42))

# Fit the pipeline to the images
pipeline.fit(images)

# Get cluster labels
cluster_labels = pipeline.predict(images)

# Create output directories if they don't exist
output_dir_label_0 = "/home/mwaltz/sampleImages/eyes_processed_0"
output_dir_label_1 = "/home/mwaltz/sampleImages/eyes_processed_1"
os.makedirs(output_dir_label_0, exist_ok=True)
os.makedirs(output_dir_label_1, exist_ok=True)

# Save images based on cluster labels
for filename, label in zip(filenames, cluster_labels):
    image_path = os.path.join(input_dir, filename)
    if label == 0:
        output_path = os.path.join(output_dir_label_0, filename)
    else:
        output_path = os.path.join(output_dir_label_1, filename)
    # Copy the image to the appropriate output directory
    os.rename(image_path, output_path)

print("Images saved to output directories.")
