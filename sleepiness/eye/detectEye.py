import cv2
import numpy as np
from joblib import load
from pathlib import Path
from sklearn.pipeline import Pipeline
from ultralytics import YOLO


def load_eye_model() -> YOLO:
    """Loads and returns the eye model."""

    try:
        model_path = Path("sleepiness") / "eye" / "eye_yolov8n.pt"
        eye_model = YOLO(model_path)
    except:
        raise FileNotFoundError(f"Error: Could not load the eye model.")

    print("Eye model loaded.")
    return eye_model

def load_clustering_model() -> Pipeline:
    """Loads and returns the clustering model for open-eye detection."""

    try:
        model_path = Path("sleepiness") / "eye" / "eye_clustering_model.joblib"
        clustering_model = load(model_path)
    except:
        raise FileNotFoundError(f"Error: Could not load the clustering model.")

    print("Clustering model loaded.")
    return clustering_model

def maxmin_scaling(image : np.ndarray) -> np.ndarray:
    """Applies MaxMin-scaling to a grayscale image."""
    # Make image float-type
    image = image.astype(np.float32)

    # Scale the image using min-max scaling
    min_val = np.min(image)
    max_val = np.max(image)
    scaled_image = (image - min_val) / (max_val - min_val) * 255

    # Convert back to uint8 and return
    return scaled_image.astype(np.uint8)

def preprocess_eye_img(img : np.ndarray)-> np.ndarray:
    # Resize the image
    img = cv2.resize(img, (50, 20))

    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # MaxMin scaling
    img = maxmin_scaling(img)

    # Flatten img
    return img.flatten()
