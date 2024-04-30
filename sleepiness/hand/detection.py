"""
Hand Landmarks detection using google's mediapipe library
"""
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sleepiness import __path__ as assetPath

def load_model(hand_model_confidence=0.15):
    """
    Load google's mediapipe hand landmarks model
    """
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

def detect_from_numpy_array(detector, image: np.ndarray):
    image = image.astype(np.uint8)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    res = detector.detect(image)
    return res if res.hand_landmarks else None