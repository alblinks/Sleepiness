"""
This file includes the full pipeline of detecting whether an aircraft seat is empty, 
contains a sleeping person, or contains a person being awake.

Authors: Martin Waltz, Niklas Paulig
"""
import os
import cv2
import numpy as np
import supervision as spv
from pathlib import Path
from joblib import load
from ultralytics import YOLO
from sklearn.pipeline import Pipeline

from sleepiness.face.detectFace import load_face_model, detect_face
from sleepiness.eye.detectEye import load_eye_model, load_clustering_model, preprocess_eye_img
from sleepiness.hand.detectHand import HandYOLO, load_hand_model


def seat_empty(img : np.ndarray) -> bool:
    """Returns 'True' if seat is empty, 'False' otherwise."""
    return False

def crop_vertically(img: np.ndarray) -> np.ndarray:
    """
    Crops the lower 20% of an image.
    
    Args:
        img (np.ndarray): The input image.
        
    Returns:
        np.ndarray: The cropped image.
    """
    height, width = img.shape[:2]  
    cropped_height = int(height * 0.8)  
    return img[:cropped_height, :]

def crop_horizontally(img: np.ndarray) -> np.ndarray:
    """
    Keeps only the middle 50% of an image (horizontally).
    
    Args:
        image (PIL.Image): The input image.
        
    Returns:
        PIL.Image: The cropped image.
    """
    height, width = img.shape[:2] 
    xmin = int(width * 0.25)
    xmax = int(width * 0.75)
    return img[:, xmin:xmax]

def detect_hands(img : np.ndarray, hand_model : HandYOLO) -> bool:
    """Detects hands in an image.
    
    Returns 'True' if at least one hand is detected with reasonable confidence.
    """ 
    # Inference
    width, height, inference_time, results = hand_model.inference(img)
    
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

def eye_detection(faceImg : np.ndarray, eye_model : YOLO) -> list:
    """Processes an image and tries to detect eyes. 
    
    Returns a list of eye regions (np.ndarrays). 
    If there are no eyes, the list will be empty."""
    # Inference
    result = eye_model(faceImg, agnostic_nms=True, verbose=False)[0]
    detections = spv.Detections.from_yolov8(result)

    # Keep only those detections associated with eyes
    eye_regions = []

    for detection in detections:

        # Class index of eyes is 0
        if detection[2] == 0:
            x_min, y_min, x_max, y_max = detection[0]
            eye_regions.append(faceImg[int(y_min):int(y_max), int(x_min):int(x_max)])
    return eye_regions

def open_eye_detection(eye_regions : list, clustering_model : Pipeline) -> bool:
    """Classifies a list of eye regions (np.ndarrays) as open- or closed-eye 
    building on a clustering model (PCA + kmeans).
    
    Returns 'True' if an open-eye is detected; False otherwise."""
    # Preprocessing
    eye_regions = [preprocess_eye_img(img) for img in eye_regions]
    labels = clustering_model.predict(eye_regions)
    
    # open eyes are cluster 0
    for l in labels:
        if l == 0:
            return True
    return False

def classify_img(path_to_img : str, 
                 face_model : YOLO, 
                 eye_model : YOLO, 
                 clustering_model : Pipeline, 
                 hand_model : HandYOLO,
                 verbose : bool = True) -> str:
    """Processes the image. 
    Returns: 
        str; element of ["not there", "awake", "sleeping"]
    """
    # Read image
    img = cv2.imread(path_to_img)

    # 1. Step: Detect whether seat is empty
    empty = seat_empty(img)

    if empty:
        return "not there"
    if verbose:
        print("Seat is not empty.")

    # 2. Step: If someone is there, detect face and select the one with largest bounding box
    face_detected, faceImg = detect_face(img=img, face_model=face_model)

    # 3. Step: Run open-eye detection on the face
    if face_detected:

        if verbose:
            print("Face detected.")
        eye_regions = eye_detection(faceImg=faceImg, eye_model=eye_model)

        if len(eye_regions) > 0:

            if verbose:
                print(f"{len(eye_regions)} eye/s detected.")
            open_eyes_detected = open_eye_detection(eye_regions=eye_regions, clustering_model=clustering_model)

            if open_eyes_detected:
        
                if verbose:
                    print("Open eye detected.")
                return "awake"

            if verbose:
                print("All eyes closed.")
        elif verbose:
            print("No eyes detected.")

    # 4. Step: If no open-eyes are detected, cut image and look for hands
    croppedImg = crop_horizontally(crop_vertically(img))

    hands_detected = detect_hands(img=croppedImg, hand_model=hand_model)

    if hands_detected:
        return "awake"
    if verbose:
        print("No hands detected in cropped image.")
    
    # 5. Step: If none of the above situations appear, we assume the person sleeps
    return "sleeping"


def main(img_folder : str, face_model : YOLO, eye_model : YOLO, clustering_model : Pipeline, hand_model : HandYOLO) -> str:
    awake_cnt = 0
    sleep_cnt = 0
    empty_cnt = 0
    N = 0

    for i, filename in enumerate(os.listdir(img_folder)):
        if i % 5 == 0:
            print(f"{i} images classified.")

        output = classify_img(path_to_img=img_folder + "/" + filename, 
                              face_model=face_model, 
                              eye_model=eye_model,
                              clustering_model=clustering_model,
                              hand_model=hand_model,
                              verbose=False)
        assert output in ["awake", "sleeping", "not there"]

        if output == "awake":
            awake_cnt += 1
        elif output == "sleeping":
            sleep_cnt += 1
        else:
            empty_cnt += 1
        N += 1

        if i == 99:
            break

    print(f"Awake:    {awake_cnt} of {N} images.")
    print(f"Sleeping: {sleep_cnt} of {N} images.")
    print(f"Empty:    {empty_cnt} of {N} images.")


if __name__ == "__main__":

    # Load models
    face_model       = load_face_model()
    eye_model        = load_eye_model()
    clustering_model = load_clustering_model()
    hand_model       = load_hand_model()
    print("------------------------------------------")

    # Perform detection
    #path_to_img = "/home/mwaltz/train/subject_0001_bright/awake/fram0015.jpg"

    #print(classify_img(path_to_img      = path_to_img, 
    #                   face_model       = face_model, 
    #                   eye_model        = eye_model, 
    #                   clustering_model = clustering_model,
    #                   hand_model       = hand_model))
    main(img_folder="/home/mwaltz/train_awake/subject_0019_bright",
         face_model=face_model, eye_model=eye_model, clustering_model=clustering_model, hand_model=hand_model)
