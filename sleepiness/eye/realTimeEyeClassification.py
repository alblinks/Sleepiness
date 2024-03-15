import cv2
import supervision as spv
import numpy as np

from sleepiness.face.detectFace import load_face_model, detect_face
import sleepiness.eye as eye
from sleepiness.main import open_eye_classify


def viz_pipeline(original_img : np.ndarray, face_xxyy : tuple, eyes_xxyy : list, eye_labels : list) -> None:
    """Displays the whole classification pipeline by drawing bounding boxes 
    of relevant features on the original image."""
    
    # Copy the original image to avoid modifying it directly
    img_with_boxes = original_img.copy()

    # Draw face bounding box
    if face_xxyy is not None:
        cv2.rectangle(img_with_boxes, (face_xxyy[0], face_xxyy[2]), (face_xxyy[1], face_xxyy[3]), (0, 255, 0), 2)

    # Draw bounding boxes for eyes
    for eye_xxyy, label in zip(eyes_xxyy, eye_labels):
        
        # Consider the eye coordinates are for face img
        xmin = eye_xxyy[0] + face_xxyy[0]
        xmax = eye_xxyy[1] + face_xxyy[0]
        ymin = eye_xxyy[2] + face_xxyy[2]
        ymax = eye_xxyy[3] + face_xxyy[2]
        cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        # Add text label
        label_position = (xmin, ymin - 10)  # Position above the bounding box
        cv2.putText(img_with_boxes, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img_with_boxes

if __name__ == "__main__":

    # Loading the model
    face_model = load_face_model()
    eye_model  = eye.load_model()
    eye_classifier = eye.load_classifier_resnet()

    # Reading frames from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()

        # Detect Face
        face_detected, faceImg, face_xxyy = detect_face(img=img, face_model=face_model, with_xyxy=True)

        # Detect eyes
        if face_detected:
            eye_regions, eye_xxyy = eye.detect(faceImg=faceImg, eye_model=eye_model)

            if len(eye_regions) > 0:
                eye_labels = open_eye_classify(eye_regions=eye_regions, eye_classifier=eye_classifier)
                eye_labels = ["closed" if l == 0 else "open" for l in eye_labels]

        # Viz
        out = viz_pipeline(original_img=img, face_xxyy=face_xxyy, eyes_xxyy=eye_xxyy, eye_labels=eye_labels)

        # Display the image
        cv2.imshow("Eye detections", out)

        # Exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
