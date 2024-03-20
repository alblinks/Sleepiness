import cv2
import numpy as np
import torch
import supervision as spv

from torch import Tensor
from torchvision import models
from joblib import load
from pathlib import Path
from sklearn.pipeline import Pipeline
from ultralytics import YOLO
from sleepiness import __path__ as p

from sleepiness.eye.FFNN.model import FFNN
from sleepiness.eye.FFNN.weights import __path__ as ffnn_WeightPath
from sleepiness.eye.CNN.model import CustomCNN
from sleepiness.eye.CNN.weights import __path__ as cnn_WeightPath


def load_model() -> YOLO:
    """Loads and returns the eye model."""

    try:
        model_path = Path(p[0]) / "eye" / "eye_yolov8n.pt"
        eye_model = YOLO(model_path)
    except:
        raise FileNotFoundError(f"Error: Could not load the eye model.")

    print("Eye model loaded.")
    return eye_model

def load_clustering_model() -> Pipeline:
    """Loads and returns the clustering model for open-eye detection."""

    try:
        model_path = Path(p[0]) / "eye" / "eye_clustering_model.joblib"
        clustering_model = load(model_path)
    except:
        raise FileNotFoundError(f"Error: Could not load the clustering model.")

    print("Clustering model loaded.")
    return clustering_model

def load_classifier_resnet() -> models.ResNet:
    """Loads and returns the ResNet18 model for open-eye detection."""

    try:
        model_path = Path(p[0]) / "eye" / "eye_classifier.pt"
        model: models.ResNet = torch.load(model_path)
    except:
        raise FileNotFoundError(f"Error: Could not load the eye classification model.")

    print("Eye classification model loaded.")
    return model

def load_classifier_ffnn() -> torch.nn.Module:
    """Loads and returns the FFNN model for open-eye detection."""

    try:
        model_path = Path(ffnn_WeightPath[0]) / "eye_epoch_26.pt"
        model: torch.nn.Module = torch.load(model_path,map_location=torch.device('cpu'))
    except Exception as e:
        raise FileNotFoundError(
            f"Error: Could not load the eye classification model.",e
        )

    print("Feed-Forward eye classification model loaded.")
    model.eval()
    model.to("cpu")
    return model

def load_classifier_cnn() -> torch.nn.Module:
    """Loads and returns the CNN model for open-eye detection."""

    try:
        model_path = Path(cnn_WeightPath[0]) / "eye_epoch_13.pt"
        model: torch.nn.Module = torch.load(model_path, map_location=torch.device('cpu'))
    except Exception as e:
        raise FileNotFoundError(
            f"Error: Could not load the eye classification model.", e
        )

    print("Convolutional eye classification model loaded.")
    model.eval()
    model.to("cpu")
    return model

def detect(faceImg : np.ndarray, eye_model : YOLO, confidence: float = 0.5) -> tuple:
    """Processes an image and tries to detect eyes. 
    
    Returns a 2-tuple:
        list of eye regions (np.ndarrays), list of bounding boxes (tuples) 
    If there are no eyes, the list will be empty."""
    # Rescale face image
    faceImg = maxmin_scaling(faceImg)

    # Inference
    result = eye_model(faceImg, agnostic_nms=True, verbose=False, conf=confidence)[0]
    detections = spv.Detections.from_yolov8(result)

    # Keep only those detections associated with eyes
    eye_regions = []
    eye_xxyy = []

    for detection in detections:

        # Class index of eyes is 0
        if detection[2] == 0:
            x_min, y_min, x_max, y_max = detection[0]
            eye_regions.append(faceImg[int(y_min):int(y_max), int(x_min):int(x_max)])
            eye_xxyy.append((int(x_min), int(x_max), int(y_min), int(y_max)))
    return eye_regions, eye_xxyy

def maxmin_scaling(image : np.ndarray) -> np.ndarray:
    """Normalizes an image to [0, 255]."""
    # Make image float-type
    image = image.astype(np.float32)

    # Scale the image using min-max scaling
    min_val = np.min(image)
    max_val = np.max(image)
    scaled_image = (image - min_val) / (max_val - min_val) * 255

    # Convert back to uint8 and return
    return scaled_image.astype(np.uint8)

def preprocess_img(img : np.ndarray)-> np.ndarray:
    # Resize the image
    img = cv2.resize(img, (50, 20))

    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # MaxMin scaling
    img = maxmin_scaling(img)

    # Flatten img
    return img.flatten()