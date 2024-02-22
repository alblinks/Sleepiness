"""
This file includes the full pipeline of detecting whether an aircraft seat is empty, 
contains a sleeping person, or contains a person being awake.

Authors: Martin Waltz, Niklas Paulig
"""

from ultralytics import YOLO
from hand.handYolo import HandYOLO
from hand.detectHand import load_hand_model, hand_inference
import cv2
import numpy as np
from PIL import Image


def load_models() -> tuple:
    """Loads the required hand and face models.
    Returns: Tuple [face_model, hand_model]"""

    face_model = YOLO("weights/yolov8n-face.pt")
    print("Face model loaded.")
    return face_model, hand_model

def seat_empty(pimg : Image) -> bool:
    """Returns 'True' if seat is empty, 'False' otherwise."""
    return False

def crop_vertically(img: Image) -> Image:
    """
    Crops the lower 20% of an image.
    
    Args:
        image (PIL.Image): The input image.
        
    Returns:
        PIL.Image: The cropped image.
    """
    width, height = img.size
    cropped_height = int(height * 0.8)  # Calculate the height of the upper 80%
    return img.crop((0, 0, width, cropped_height)) # left, top, right, bottom

def crop_horizontally(img: Image) -> Image:
    """
    Keeps only the middle 50% of an image (horizontally).
    
    Args:
        image (PIL.Image): The input image.
        
    Returns:
        PIL.Image: The cropped image.
    """
    width, height = img.size
    width_left  = int(width * 0.25)
    width_right = int(width * 0.75)
    return img.crop((width_left, 0, width_right, height)) # left, top, right, bottom

def detect_face(pimg : Image, face_model : YOLO) -> tuple:
    """Detects faces on an image.
    
    Returns: 
        Tuple of (bool, Image). 
        
    The bool is 'True' if at least one face is detected. 
    The Image is then the (reduced-size) image containing the face with the largest bounding box, otherwise the original image. 
    """ 
    results = face_model.predict(pimg, stream=False)[0]

    # No image detected
    if len(results.boxes) == 0:
        return False, pimg

    # Select image with largest bounding box
    largest_face_area  = 0
    largest_face_image = pimg  # Default to original image
    
    for box in results.boxes.xyxy.cpu().numpy():

        # Calculate face area
        face_area = (box[2] - box[0]) * (box[3] - box[1])

        # Crop the original image using the largest face bounding box
        if face_area > largest_face_area:
            largest_face_area  = face_area                   
            largest_face_image = pimg.crop((box[0], box[1], box[2], box[3]))

    return True, largest_face_image

def detect_hands(pimg : Image, hand_model : HandYOLO) -> bool:
    """Detects hands in an image.
    
    Returns 'True' if at least one hand is detected with reasonable confidence.
    """ 
    # Inference
    width, height, inference_time, results = hand_inference(pimg=pimg, hand_model=hand_model)
    
    # How many hands should be shown
    hand_count = len(results)

    # Testing: Display hands
    #for detection in results[:hand_count]:
    #    id, name, confidence, x, y, w, h = detection

        # draw a bounding box rectangle and label on the image
    #    color = (0, 255, 255)
    #    cv2.rectangle(npImg, (x, y), (x + w, y + h), color, 2)
    #    text = "%s (%s)" % (name, round(confidence, 2))
    #    cv2.putText(npImg, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
    #                0.5, color, 2)
    #cv2.namedWindow("preview")
    #cv2.imshow("preview", npImg)
    #cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    #cv2.destroyAllWindows()

    if hand_count == 0:
        return False
    else:
        return True

def detect(path_to_img : str, face_model : YOLO, hand_model : HandYOLO) -> str:
    """Processes the image. 
    Returns: 
        str; element of ["not there", "awake", "sleeping"]
    """
    # Read image
    pimg = Image.open(path_to_img)

    # 1. Step: Detect whether seat is empty
    empty = seat_empty(pimg)

    if empty:
        return "not there"

    # 2. Step: If someone is there, detect face and select the one with largest bounding box
    face_detected, faceImg = detect_face(pimg=pimg, face_model=face_model)

    # 3. Step: Run open-eye detection on the face
    if face_detected:
        open_eye_detected = False

        # TBF !!

        if open_eye_detected:
            return "awake"
    
    # 4. Step: If no open-eyes are detected, cut image and look for hands
    croppedImg = crop_horizontally(crop_vertically(pimg))

    hands_detected = detect_hands(pimg=croppedImg, hand_model=hand_model)

    if hands_detected:
        return "awake"
    
    # 5. Step: If none of the above situations appear, we assume the person sleeps
    return "sleeping"


if __name__ == "__main__":
    # Load models
    face_model, hand_model = load_models()

    # Perform detection
    path_to_img = "/home/mwaltz/LGCS_Daten_Video_Teil1_gelabelt/subject_0001_bright/awake/fram0001.jpg"
    print(detect(path_to_img=path_to_img, face_model=face_model, hand_model=hand_model))
