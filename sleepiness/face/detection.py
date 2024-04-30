import base64
from ultralytics import YOLO
import numpy as np
from pathlib import Path
from sleepiness import __path__ as p
from sleepiness.utility.logger import logger
from sleepiness.utility.misc import download_file_with_progress

FMP = """aHR0cHM6Ly93d3cuZHJ
vcGJveC5jb20vc2NsL2ZpL3V1bDZ
6cmUxZTJsMmttd29hZTFnbC95b2x
vdjhuLWZhY2UucHQ/cmxrZXk9MmU
5bzc4b3VyNHp3enF3azlxNmZmZHh
keCZzdD1ndHhtbGFqeiZkbD0x"""


def load_model() -> YOLO:
    """Loads and returns the face model."""

    model_path = Path(p[0]) / "face" / "yolov8n-face.pt"
    if not model_path.exists():
        logger.info("Face model not found. Downloading...")
        download_file_with_progress(
            base64.b64decode(FMP).decode(), model_path,
            "Downloading the face detection model..."
        )
    face_model = YOLO(model_path)
    # except:
    #     raise FileNotFoundError("Error: Could not load the face model. Check the paths.")

    print("Face model loaded.")
    return face_model

def detect(img : np.ndarray, face_model : YOLO, with_xyxy : bool = False) -> tuple:
    """Detects faces on an image.
    
    Returns: 
        Tuple of (bool, Image). If 'with_xyxy', the bounding box as a tuple (xmin, xmax, ymin, ymax) is returned.
        
    The bool is 'True' if at least one face is detected. 
    The Image is then the (reduced-size) image containing the face with the largest bounding box, otherwise the original image. 
    """ 
    results = face_model.predict(img, stream=False, verbose=False)[0]

    # No image detected
    if len(results.boxes) == 0:
        if with_xyxy:
            return False, img, None
        else:
            return False, img

    # Select image with largest bounding box
    largest_face_area  = 0
    largest_face_image = img  # Default to original image
    largest_face_xxyy = None
    
    for box in results.boxes.xyxy.cpu().numpy():

        # Selection
        xmin = int(box[0])
        xmax = int(box[2])
        ymin = int(box[1])
        ymax = int(box[3])

        # Calculate face area
        face_area = (xmax - xmin) * (ymax - ymin)

        # Crop the original image using the largest face bounding box
        if face_area > largest_face_area:
            largest_face_area  = face_area
            largest_face_image = img[ymin:ymax, xmin:xmax]
            largest_face_xxyy = (xmin, xmax, ymin, ymax)

    if with_xyxy:
        return True, largest_face_image, largest_face_xxyy
    return True, largest_face_image
