"""
This file includes the full pipeline of detecting whether an aircraft seat is empty, 
contains a sleeping person, or contains a person being awake.

Authors: Martin Waltz, Niklas Paulig
"""
import os
import cv2
import numpy as np
import supervision as spv
import torch

from PIL import Image
from torchvision import models
from ultralytics import YOLO
from sklearn.pipeline import Pipeline
from torchvision.transforms import transforms

from sleepiness.face.detectFace import load_face_model, detect_face
from sleepiness.eye.detectEye import (load_eye_model, load_clustering_model, preprocess_eye_img, 
                                      load_eye_classifier, max_min_scaling_t, maxmin_scaling)
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

def detect_hands(img : np.ndarray, hand_model : HandYOLO) -> tuple:
    """Detects hands in an image.
    
    Returns tuple of bool and a list.
    The bool is 'True' if at least one hand is detected with reasonable confidence.
    The list contains tuples of (xmin, xmax, ymin, ymax) bounding boxes of the detected hands.
    """ 
    # Inference
    width, height, inference_time, results = hand_model.inference(img)
    
    # How many hands should be shown
    hand_count = len(results)

    hand_xxyy = []
    for r in results:
        id, name, confidence, x, y, w, h = r
        hand_xxyy.append((x, x+w, y, y+h))

    # Testing: Display hands
    #for detection in results[:hand_count]:
    #    id, name, confidence, x, y, w, h = detection

        # draw a bounding box rectangle and label on the image
    #    color = (0, 255, 255)
    #    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    #    text = "%s (%s)" % (name, round(confidence, 2))
    #    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
    #                0.5, color, 2)
    #cv2.namedWindow("preview")
    #cv2.imshow("preview", img)
    #cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    #cv2.destroyAllWindows()

    if hand_count == 0:
        return False, hand_xxyy
    else:
        return True, hand_xxyy

def eye_detection(faceImg : np.ndarray, eye_model : YOLO) -> tuple:
    """Processes an image and tries to detect eyes. 
    
    Returns a 2-tuple:
        list of eye regions (np.ndarrays), list of bounding boxes (tuples) 
    If there are no eyes, the list will be empty."""
    # Rescale face image
    faceImg = maxmin_scaling(faceImg)

    # Inference
    result = eye_model(faceImg, agnostic_nms=True, verbose=False)[0]
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

def open_eye_clustering(eye_regions : list, clustering_model : Pipeline) -> bool:
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

def open_eye_resnet(eye_regions : list, eye_classifier : models.ResNet) -> bool:
    """Classifies a list of eye regions (np.ndarrays) as open- or closed-eye using a ResNet classifier.
    
    Returns 'True' if an open-eye is detected; False otherwise."""
    transform = transforms.Compose([
        transforms.Resize((50,20)),
        transforms.ToTensor(),
        max_min_scaling_t,
    ])
    labels = []
    for r in eye_regions:

        # Convert the NumPy array to a PIL image with mode 'RGB'
        pil_img = Image.fromarray(r, mode='RGB')

        # Torch transform
        torch_img = transform(pil_img).unsqueeze(0)

        # Forward
        logprobs = eye_classifier(torch_img)

        # Open eyes have label 1
        labels.append(torch.argmax(logprobs).item())
    return labels

def transform_xxyy_for_cropped_img(full_img : np.ndarray, xxyy : tuple, keep_horizontal : float = 0.5):
    """Computes the bounding box coordinates (xxyy) for the full image for a given bounding box (xxyy) of a cropped img.
    Cropping means keep only the middle 'keep_horizontal' percent of pixels of an image.
    
    Args:
        full_img: Full size image.
        xxyy: Bounding box coordinates of the cropped image.
        keep_horizontal: Percentage of horizontal cropping.
    Returns:
        Bounding box coordinates for the full img.
    """
    # Img size
    full_height, full_width = full_img.shape[:2] 

    # Unpack the bounding box coordinates
    x_min, x_max, y_min, y_max = xxyy
       
    # Calculate the horizontal and vertical offsets based on cropping percentages
    x_off = full_width * (1 - keep_horizontal) / 2
    y_off = 0
    return (int(x_min + x_off), int(x_max + x_off), int(y_min + y_off), int(y_max + y_off))

def viz_pipeline(original_img : np.ndarray, face_xxyy : tuple, eyes_xxyy : list, hands_xxyy : list, label : str, text : str) -> None:
    """Displays the whole classification pipeline by drawing bounding boxes 
    of relevant features on the original image."""
    
    # Copy the original image to avoid modifying it directly
    img_with_boxes = original_img.copy()

    # Draw face bounding box
    if face_xxyy is not None:
        cv2.rectangle(img_with_boxes, (face_xxyy[0], face_xxyy[2]), (face_xxyy[1], face_xxyy[3]), (0, 255, 0), 2)

    # Draw bounding boxes for eyes
    for eye_xxyy in eyes_xxyy:
        
        # Consider the eye coordinates are for face img
        xmin = eye_xxyy[0] + face_xxyy[0]
        xmax = eye_xxyy[1] + face_xxyy[0]
        ymin = eye_xxyy[2] + face_xxyy[2]
        ymax = eye_xxyy[3] + face_xxyy[2]
        cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    # Draw bounding boxes for hands
    for hand_xxyy in hands_xxyy:
        hand_xxyy = transform_xxyy_for_cropped_img(full_img=original_img, xxyy=hand_xxyy)
        cv2.rectangle(img_with_boxes, (hand_xxyy[0], hand_xxyy[2]), (hand_xxyy[1], hand_xxyy[3]), (255, 0, 0), 2)

    # Write some text with line breaks
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (20, 15)  # Position of the text
    fontScale = 0.5
    color = (0, 0, 0)
    thickness = 1
    line_type = cv2.LINE_AA

    # Split the text into lines
    lines = text.split('\n')

    # Write each line of text with appropriate line spacing
    for i, line in enumerate(lines):
        y = org[1] + i * 20  # Adjust spacing between lines (you can modify this value)
        cv2.putText(img_with_boxes, line, (org[0], y), font, fontScale, color, thickness, line_type)

    # Concatenate the original image and the image with bounding boxes horizontally
    combined_img = np.hstack((original_img, img_with_boxes))

    # Display the image with bounding boxes
    #cv2.imshow("Estimate: " + label, combined_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Save the image with bounding boxes
    output_file = "ex4_" + label + ".jpg"  # Change the filename and extension as needed
    cv2.imwrite(output_file, combined_img)

def classify_img(path_to_img : str, 
                 face_model : YOLO, 
                 eye_model : YOLO, 
                 clustering_model : Pipeline, 
                 eye_classifier : models.ResNet,
                 hand_model : HandYOLO,
                 viz : bool = False) -> str:
    """Processes the image. 
    Returns: 
        str; element of ["not there", "awake", "sleeping"]
    """

    # Default
    out = "sleeping"
    s = ""

    # Read image
    img = cv2.imread(path_to_img)
    assert img is not None, "Could not load the image."

    # 1. Step: Detect whether seat is empty
    empty = seat_empty(img)

    if empty:
        out = "not there"
        if not viz:
            return out
    if viz:
        s += "Seat is not empty.\n"

    # 2. Step: If someone is there, detect face and select the one with largest bounding box
    face_detected, faceImg, face_xxyy = detect_face(img=img, face_model=face_model, with_xyxy=True)

    # 3. Step: Run open-eye detection on the face
    if face_detected:

        if viz:
            s += "Face detected.\n"
        eye_regions, eye_xxyy = eye_detection(faceImg=faceImg, eye_model=eye_model)

        if len(eye_regions) > 0:

            if viz:
                s += f"{len(eye_regions)} eye/s detected.\n"

            #open_eyes_detected = open_eye_clustering(eye_regions=eye_regions, clustering_model=clustering_model)
            eye_labels = open_eye_resnet(eye_regions=eye_regions, eye_classifier=eye_classifier)

            if any(eye_labels):
        
                if viz:
                    s += f"{sum(eye_labels)} open. {len(eye_labels)-sum(eye_labels)} closed. \n"
                out = "awake"
                if not viz:
                    return "awake"
            elif viz:
                s += "All eyes closed.\n"

        elif viz:
            s += "No eyes detected.\n"

    # 4. Step: If no open-eyes are detected, cut image and look for hands
    croppedImg = crop_horizontally(crop_vertically(img))

    hands_detected, hands_xxyy = detect_hands(img=croppedImg, hand_model=hand_model)

    if hands_detected:
        if viz:
            s += "Hand/s detected in cropped image.\n"
        out = "awake"
        if not viz:
            return out
    elif viz:
        s += "No hands detected in cropped image.\n"
    
    # 5. Step: If none of the above situations appear, we assume the person sleeps
    if viz:
        viz_pipeline(original_img=img, face_xxyy=face_xxyy, eyes_xxyy=eye_xxyy, hands_xxyy=hands_xxyy, label=out, text=s)
    return out


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
                              viz=False)
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
    #clustering_model = load_clustering_model()
    eye_classifier   = load_eye_classifier()
    hand_model       = load_hand_model()
    print("------------------------------------------")

    # Perform detection
    #path_to_img = "/home/mwaltz/train/subject_0025_bright/awake/fram0106.jpg" # ex1
    #path_to_img = "/home/mwaltz/train_awake/subject_0019_bright/fram3501.jpg" # ex2
    #path_to_img = "/home/mwaltz/train_awake/subject_0012_bright/fram0084.jpg" # ex3
    path_to_img = "/home/mwaltz/train/subject_0022_dimmed/sleeping/fram2139.jpg" # ex4

    print(classify_img(path_to_img      = path_to_img, 
                       face_model       = face_model, 
                       eye_model        = eye_model, 
                       clustering_model = None,
                       eye_classifier   = eye_classifier,
                       hand_model       = hand_model,
                       viz = True))
    #main(img_folder="/home/mwaltz/train_awake/subject_0019_bright",
    #     face_model=face_model, eye_model=eye_model, clustering_model=clustering_model, hand_model=hand_model)
