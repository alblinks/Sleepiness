import torch
import numpy as np
from PIL import Image
from pathlib import Path
from .weights import __path__ as p
from .transforms import val_transform
from sleepiness.face.CNN.model import CustomCNN


def load_model() -> torch.nn.Module:
    """Loads and returns the face model."""

    try:
        model_path = Path(p[0]) / "face_epoch_1.pt"
        face_model = torch.load(model_path)
    except:
        raise FileNotFoundError(
            "Error: Could not load the face classification model. Check the paths."
        )

    print("Face detection model loaded.")
    face_model.eval()
    return face_model

def classify(img : np.ndarray, face_model : torch.nn.Module) -> int:
    """Classifies the face in the image as awake or sleepy.
    
    Returns: 
        int:
            - 0: Awake
            - 1: Sleepy
    """ 
    pil_img = Image.fromarray(img, mode="RGB")
    timg = val_transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        logps = face_model(timg)
        return torch.argmax(logps).item()
