import os
import cv2
import numpy as np

from joblib import dump
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sleepiness.eye import preprocess_img


def load_images(input_dir):
    """Function to load and preprocess images."""
    images = []
    filenames = []

    for i, filename in enumerate(os.listdir(input_dir)):
        if i % 100 == 0:
            print(i)

        if i < 10_000:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                
                # Construct path
                image_path = os.path.join(input_dir, filename)

                # Read image
                img = cv2.imread(image_path)

                # Preprocess and store image
                images.append(preprocess_img(img))

                # Keep track of filenames
                filenames.append(filename)
    return np.array(images), filenames


if __name__ == "__main__":

    # Input directory containing eye images
    input_dir = "/home/mwaltz/sampleImages/eyes"

    # Output directory for clustering results
    output_dir_label_0 = "/home/mwaltz/sampleImages/cluster0"
    output_dir_label_1 = "/home/mwaltz/sampleImages/cluster1"

    # Output directory for clustering model
    output_dir_model = "/home/mwaltz/sampleImages/cluster_model"

    # Create output directories if they don't exist
    os.makedirs(output_dir_label_0, exist_ok=True)
    os.makedirs(output_dir_label_1, exist_ok=True)
    os.makedirs(output_dir_model, exist_ok=True)

    # Load images
    images, filenames = load_images(input_dir)
    print("Image loading completed.")

    # Define the number of clusters
    num_clusters = 2

    # Define the clustering pipeline
    pipeline = make_pipeline(PCA(n_components=10), 
                             KMeans(n_clusters=num_clusters, random_state=42))

    # Fit the pipeline to the images
    pipeline.fit(images)
    print("Clustering completed.")

    # Save the clustering model
    model_filename = "clustering_model.joblib"
    dump(pipeline, output_dir_model + "/" + model_filename)

    # Get cluster labels
    cluster_labels = pipeline.predict(images)

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
