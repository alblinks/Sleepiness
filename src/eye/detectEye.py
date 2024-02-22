from ultralytics import YOLO


def load_eye_model() -> YOLO:
    """Loads and returns the eye model."""

    try:
        eye_model = YOLO("eye_yolov8n.pt")
    except:
        raise FileNotFoundError("Error: Could not load the eye model. Check the paths.")

    print("Eye model loaded.")
    return eye_model
