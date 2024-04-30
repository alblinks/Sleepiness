"""
This file includes the full pipeline of detecting whether an aircraft seat is empty, 
contains a sleeping person, or contains a person being awake.

Authors: Martin Waltz, Niklas Paulig
"""
import cv2
import pickle
import numpy as np
import torch
import uuid

from PIL import Image
from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from torchvision.transforms import transforms

import sleepiness.hand.detection as hand

from sleepiness import PassengerState

import sleepiness.face.yoloface as facedetect
import sleepiness.eye as eye
import sleepiness.hand as hand
from sleepiness.empty_seat.pixdiff import (
    is_empty, preprocess as empty_preprocessor
)
# Load the average pixel map
from sleepiness.empty_seat.pixdiff import __path__ as pixdiff_path 
with open(f"{pixdiff_path[0]}/avgmap.nparray", "rb") as f:
    AVGMAP = pickle.load(f)
    
def crop_vertically(img: np.ndarray) -> np.ndarray:
    """
    Crops the lower 20% of an image.
    Seat cam is 40%.
    
    Args:
        img (np.ndarray): The input image.
        
    Returns:
        np.ndarray: The cropped image.
    """
    height, width = img.shape[:2]  
    cropped_height = int(height * 0.6)  
    return img[:cropped_height, :]

def crop_horizontally(img: np.ndarray) -> np.ndarray:
    """
    Keeps only the middle 50% of an image (horizontally).
    Seat cam is 100%
    
    Args:
        image (PIL.Image): The input image.
        
    Returns:
        PIL.Image: The cropped image.
    """
    return img
    height, width = img.shape[:2] 
    xmin = int(width * 0.25)
    xmax = int(width * 0.75)
    return img[:, xmin:xmax]
    
def crop_image(img: np.ndarray, xmin: int, xmax: int, ymin: int, ymax: int) -> np.ndarray:
    """
    Crops an image based on the bounding box coordinates in percentage.
    
    Args:
        img (np.ndarray): The input image.
        xmin (int): The minimum x-coordinate.
        xmax (int): The maximum x-coordinate.
        ymin (int): The minimum y-coordinate.
        ymax (int): The maximum y-coordinate.
        
    Example:
        # Crop the center 50% of the image
        crop_image(img, 0.25, 0.75, 0.25, 0.75)
        
    Returns:
        np.ndarray: The cropped image.
    """
    
    height, width = img.shape[:2] 
    x_min = int(width * xmin)
    x_max = int(width * xmax)
    y_min = int(height * ymin)
    y_max = int(height * ymax)
    return img[y_min:y_max, x_min:x_max]

class Pipeline(ABC):
    """
    Abstract base class for a pipeline logic used to 
    clssify images of airplane passengers into
    one of three states given by `sleepiness.PassengerState`:
        - AWAKE
        - SLEEPING
        - NOTTHERE
    
    The pipeline is executed by calling the `classify`
    method with a path to an image as input.
    """
    
    @abstractmethod
    def classify(self, img_path : str) -> PassengerState:
        """
        Classifies the image using the pipeline logic.
        
        Args:
            img_path: Path to the image.
        
        Returns:
            PassengerState: The classification result.
        """
        raise NotImplementedError

class FullPipeline(Pipeline):

    def __init__(self,
                 eye_model_confidence : float,
                 hand_model_confidence : float,
                 hand_model_crop : list[float,float,float,float],):
        
        """
        Args:
            eye_model_confidence: Confidence threshold for eye detection.
            hand_model_confidence: Confidence threshold for hand detection.
            hand_model_crop: Bounding box coordinates for cropping the image for hand detection
                as [xmin, xmax, ymin, ymax] in percentage.
        """
        assert 0 <= eye_model_confidence <= 1, "Confidence must be between 0 and 1."
        assert 0 <= hand_model_confidence <= 1, "Confidence must be between 0 and 1."
        assert len(hand_model_crop) == 4 and all([0 <= x <= 1 for x in hand_model_crop]),\
            "Bounding box must be in percentage."
        
        self.hand_model_crop = hand_model_crop
        self.face_model = facedetect.load_model()
        self.eye_model = eye.load_model()
        self.eye_classifier = eye.load_classifier_cnn()
        self.hand_model = hand.load_model(hand_model_confidence)
        
        self.eye_model_confidence = eye_model_confidence

    def open_eye_clustering(self, eye_regions : list, clustering_model : Pipeline) -> bool:
        """Classifies a list of eye regions (np.ndarrays) as open- or closed-eye 
        building on a clustering model (PCA + kmeans).
        
        Returns 'True' if an open-eye is detected; False otherwise."""
        # Preprocessing
        eye_regions = [eye.preprocess_img(img) for img in eye_regions]
        labels = clustering_model.predict(eye_regions)
        
        # open eyes are cluster 0
        for l in labels:
            if l == 0:
                return True
        return False

    def open_eye_classify(self, eye_regions : list[np.ndarray], 
                        eye_classifier : torch.nn.Module
                        ) -> list[int]:
        """Classifies a list of eye regions (np.ndarrays) as open- or closed-eye using a ResNet classifier.
        
        Returns 'True' if an open-eye is detected; False otherwise."""
        transform = transforms.Compose([
            transforms.Resize((20,50)), # height, width
            transforms.ToTensor(),
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

    def transform_xxyy_for_cropped_img(self, 
                                       full_img : np.ndarray, 
                                       cropped_img : np.ndarray,
                                       xxyy : tuple
                                    )-> tuple[int,int,int,int]:
        """Computes the bounding box coordinates (xxyy) for the full image for a given bounding box (xxyy) of a cropped img.
        Cropping means keep only the middle 'keep_horizontal' percent of pixels of an image.
        
        Args:
            full_img: Full size image.
            xxyy: Bounding box coordinates in percentage for the cropped image.
            keep_horizontal: Percentage of horizontal cropping.
        Returns:
            Bounding box coordinates for the full img.
        """
        # Keep only 'keep_horizontal' percent of pixels of an image.
        # This is to account for the fact that the hand detection model 
        # used on cropped images
        keep_horizontal = self.hand_model_crop[3] - self.hand_model_crop[2]
        keep_vertical = self.hand_model_crop[1] - self.hand_model_crop[0]
        
        # Img size
        full_height, full_width = full_img.shape[:2] 
        cropped_height, cropped_width = cropped_img.shape[:2]

        # Unpack the bounding box coordinates
        px_min, px_max, py_min, py_max = xxyy
        x_min = int(px_min * cropped_width)
        x_max = int(px_max * cropped_width)
        y_min = int(py_min * cropped_height)
        y_max = int(py_max * cropped_height)
        
        # Calculate the horizontal and vertical offsets based on cropping percentages
        x_off = full_width * (1 - keep_horizontal) / 2
        y_off = full_height * (1 - keep_vertical) / 2
        return (int(x_min + x_off), int(x_max + x_off), int(y_min + y_off), int(y_max + y_off))

    def visualize(self, 
                  original_img : np.ndarray, 
                  crop_img : np.ndarray,
                  face_xxyy : tuple, eyes_xxyy : list, 
                  hands_xxyy : list, label : str, 
                  text : str) -> None:
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
            hand_xxyy = self.transform_xxyy_for_cropped_img(
                full_img=original_img, cropped_img=crop_img, xxyy=hand_xxyy
            )
            cv2.rectangle(
                img_with_boxes, 
                (hand_xxyy[0], hand_xxyy[2]), 
                (hand_xxyy[1], hand_xxyy[3]), 
                (255, 0, 0), 
                2
            )

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
        # Change the filename and extension as needed
        output_file = "full_pipeline_eval/"+ label + "_" + str(uuid.uuid1()) + ".jpg"
        cv2.imwrite(output_file, combined_img)

    def classify(self,
                img_or_path : str | np.ndarray, 
                viz : bool = True) -> PassengerState:
        """Processes the image. 
        Returns: 
            - PassengerState.AWAKE if the person is awake
            - PassengerState.SLEEPING if the person is sleeping
            - PassengerState.NOTTHERE if the seat is empty
        
        
        Args:
            path_to_img: Path to the image.
            face_model: Model for face detection.
            eye_model: Model for eye detection.
            clustering_model: Pipeline for clustering eye regions. !Currently not in use.
            eye_classifier: PyTorch model for eye classification.
            hand_model: YOLO model for hand detection.
            viz: If True, the function will display the image with bounding boxes and text.
        """

        # Default
        state = PassengerState.SLEEPING
        s = []

        # Read image
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
        else: img = img_or_path
        assert img is not None, "Could not load the image."

        # 1. Step: Detect whether seat is empty
        # TODO: switch empty detection to cv2
        if isinstance(img_or_path, str):
            proc_for_empty = empty_preprocessor(Image.open(img_or_path))
        else:
            proc_for_empty = empty_preprocessor(Image.fromarray(img))

        if is_empty(proc_for_empty ,threshold= 0.08, map=AVGMAP):
            state = PassengerState.NOTTHERE
            if not viz:
                return state
        if viz:
            s.append("Seat is not empty.")

        # 2. Step: If someone is there, detect face and select the one with largest bounding box
        face_detected, faceImg, face_xxyy = facedetect.detect(
            img=img, face_model=self.face_model, with_xyxy=True
        )

        # 3. Step: Run open-eye detection on the face
        if face_detected:

            if viz:
                s.append("Face detected.")
            eye_regions, eye_xxyy = eye.detect(
                faceImg=faceImg, eye_model=self.eye_model, confidence=self.eye_model_confidence
            )

            if len(eye_regions) > 0:

                if viz:
                    s.append(f"{len(eye_regions)} eye/s detected.")

                eye_labels = self.open_eye_classify(
                    eye_regions=eye_regions, 
                    eye_classifier=self.eye_classifier
                )

                if any(eye_labels):
            
                    if viz:
                        s.append(
                            f"{sum(eye_labels)} open. "
                            f"{len(eye_labels)-sum(eye_labels)} closed."
                        )
                    state = PassengerState.AWAKE
                    if not viz:
                        return state
                elif viz:
                    s.append("All eyes closed.")

            elif viz:
                s.append("No eyes detected.")
        else:
            eye_xxyy = []

        # 4. Step: If no open-eyes are detected, cut image and look for hands
        croppedImg = crop_image(img, *self.hand_model_crop)

        hand_detection = hand.detect_from_numpy_array(self.hand_model,croppedImg)
        if hand_detection is not None:
            hands_detected = True
            hands_xxyy = hand.get_bbox_from_landmarks(hand_detection.hand_landmarks)
        else: 
            hands_detected = False
            hands_xxyy = []

        if hands_detected:
            if viz:
                s.append("Hand/s detected in cropped image.")
            state = PassengerState.AWAKE
            if not viz:
                return state
        elif viz:
            s.append("No hands detected in cropped image.")
        
        # 5. Step: If none of the above situations appear, we assume the person sleeps
        if viz:
            self.visualize(
                original_img=img, 
                crop_img=croppedImg,
                face_xxyy=face_xxyy, 
                eyes_xxyy=eye_xxyy, 
                hands_xxyy=hands_xxyy, 
                label=state.name.lower(), 
                text="\n".join(s)
            )
        return state