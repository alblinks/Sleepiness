import cv2
import numpy as np
import torch
import base64
import supervision as spv

from pathlib import Path
from ultralytics import YOLO
from sleepiness import __path__ as p

from sleepiness.eye.CNN.model import CustomCNN
from sleepiness.eye.CNN.weights import __path__ as cnn_WeightPath
from sleepiness.utility.logger import logger
from sleepiness.utility.misc import download_file_with_progress

CLP = """aHR0cHM6Ly93d3cuZHJvcGJ
veC5jb20vc2NsL2ZpL2FxNzUyZXBzOHU
5bXFoZ3NyaTFsai9leWVfY2xhc3NpZml
lcl9jbm4ucHQ/cmxrZXk9dTBzcTBqamt
5MjE0ZnFnc3YzZ2Z5ZDk1cCZzdD10YnV
sYTJ1MSZkbD0x"""

DP = """aHR0cHM6Ly93d3cuZHJvcGJve
C5jb20vc2NsL2ZpL3h4c29wbTE5cjdnO
WNmYjIyZTBubC9leWVfeW9sb3Y4bi5wd
D9ybGtleT1tZDRuNXBpcWpucmsxNWw5a
GdncmpoZ3JhJnN0PTlvcnZvajduJmRsP
TE="""

def load_model() -> YOLO:
    """Loads and returns the eye model."""

    model_path = Path(p[0]) / "eye" / "eye_yolov8n.pt"
    if not model_path.exists():
        logger.info("Eye model not found. Downloading...")
        download_file_with_progress(
            base64.b64decode(DP).decode(), model_path,
            "Downloading the eye detection model..."
        )
    eye_model = YOLO(model_path)

    print("Eye model loaded.")
    return eye_model

def load_classifier_cnn() -> torch.nn.Module:
    """Loads and returns the CNN model for open-eye detection."""
    
    model_path = Path(cnn_WeightPath[0]) / "eye_classifier_cnn.pt"
    if not model_path.exists():
        logger.info("Eye classification model not found. Downloading...")
        download_file_with_progress(
            base64.b64decode(CLP).decode(), model_path,
            "Downloading the eye classification model..."
        )
    model: torch.nn.Module = torch.load(model_path, map_location=torch.device('cpu'))

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