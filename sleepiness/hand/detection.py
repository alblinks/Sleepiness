"""
Hand Landmarks detection using google's mediapipe library
"""
from pathlib import Path
import base64
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sleepiness import __path__ as assetPath
from sleepiness.utility.logger import logger
from sleepiness.utility.misc import download_file_with_progress

HMP = """aHR0cHM6Ly93d3cuZHJvcGJveC5jb20
vc2NsL2ZpLzZ4Z244Z2diMWMxczF0ZHI2cmYwaC9
oYW5kX2xhbmRtYXJrZXIudGFzaz9ybGtleT11NXp
senBjMm11cDdwOGwwZDUwNmF6bG51JnN0PXVqdGQ
yYmZwJmRsPTE="""

def load_model(hand_model_confidence=0.15):
    """
    Load google's mediapipe hand landmarks model
    """
    model_path = f'{assetPath[0]}/hand/hand_landmarker.task'
    if not Path(model_path).exists():
        logger.info("Hand landmarks model not found. Downloading...")
        download_file_with_progress(
            base64.b64decode(HMP).decode(), model_path,
            "Downloading the hand landmarks model..."
        )
    base_options = python.BaseOptions(
        model_asset_path=f'{assetPath[0]}/hand/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence = hand_model_confidence,
        min_hand_presence_confidence = hand_model_confidence,
        min_tracking_confidence = hand_model_confidence
    )

    return vision.HandLandmarker.create_from_options(options)

def draw_landmarks_on_image(image, result):
    annotated_image = image.copy()
    for hand_landmarks in result.hand_landmarks:
        for l1, l2 in zip(hand_landmarks,hand_landmarks[1:]):
            cv2.line(annotated_image, (int(l1.x*image.shape[1]), int(l1.y*image.shape[0])),
                     (int(l2.x*image.shape[1]), int(l2.y*image.shape[0])), (57, 17, 122), 2)
    return annotated_image

def get_bbox_from_landmarks(landmarks) -> list[float]:
    bboxes = []
    for hand_landmark in landmarks:
        x,y = np.array([(l.x, l.y) for l in hand_landmark]).T
        bboxes.append([min(x), max(x), min(y), max(y)])
    return bboxes

def detect_from_numpy_array(detector, image: np.ndarray):
    image = image.astype(np.uint8)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    res = detector.detect(image)
    return res if res.hand_landmarks else None