from handYolo import HandYOLO
import cv2
import numpy as np
from PIL import Image


def load_hand_model() -> HandYOLO:
    """Loads and returns the hand model."""

    try:
        hand_model = HandYOLO("hand/cross-hands.cfg", "hand/cross-hands.weights", ["hand"])
    except:
        raise FileNotFoundError("Error: Could not load the hand model. Check the paths.")
    
    hand_model.size = 416
    hand_model.confidence = 0.2

    print("Hand model loaded.")
    return hand_model

def hand_inference(pimg : Image, hand_model : HandYOLO) -> tuple:
    """Performs the YOLO-based hand inference for an image. Take care with RGB values!

    Returns:
        width, height, inference_time, results
    """
    npImg = cv2.cvtColor(np.array(pimg), cv2.COLOR_BGR2RGB)
    return hand_model.inference(npImg)
