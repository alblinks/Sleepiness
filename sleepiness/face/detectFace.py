from ultralytics import YOLO
from pathlib import Path    
from PIL import Image

def load_face_model() -> YOLO:
    """Loads and returns the face model."""

    try:
        face_model = YOLO("sleepiness/face/yolov8n-face.pt")
    except:
        raise FileNotFoundError("Error: Could not load the face model. Check the paths.")

    print("Face model loaded.")
    return face_model

def predict_and_save(img: Path, model: YOLO):
    """Predict and save the image 
    with bounding boxes and confidence"""
    pimg = Image.open(img)
    model.predict(pimg, save=True, project= "out", name="predict", exist_ok=True)

def predict_and_save_dir(dir: Path, model: YOLO):
    """Predict and save all 
    images in a directory"""
    for img in dir.iterdir():
        predict_and_save(img, model)
        
def predict_bbox(img: str, model: YOLO):
    """
    Predict bounding boxes and confidence for a single image
    """
    pimg = Image.open(img)
    results = model.predict(pimg, stream=True)
    for result in results:
        r = result.boxes.xyxy
        conf = result.boxes.conf
        # Draw bounding boxes and confidence
        for box,conf in zip(r.cpu().numpy(),conf.cpu().numpy()):
            if conf > 0.5:
                print(box, conf)
