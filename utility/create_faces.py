import cv2
import numpy as np
from PIL import Image
import os

from pipeline import load_models, detect_face
from ultralytics import YOLO
from handYolo import HandYOLO

def save_detected_faces(source_folder : str, destination_folder : str, face_model : YOLO) -> None:
    """Detect and save faces in new images."""

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Loop over the images in the source folder
    for filename in os.listdir(source_folder):

        if filename.endswith(".jpg"):

            # Construct the full path to the image
            image_path = os.path.join(source_folder, filename)
            
            # Load the image using PIL
            image = Image.open(image_path)
            
            # Detect the face in the image
            face_detected, face_image = detect_face(pimg=image, face_model=face_model)
            
            # If a face is detected, save it
            if face_detected:

                # Construct the filename for the face image
                face_filename = f"{os.path.splitext(filename)[0]}_face.png"

                # Construct the full path to save the face image
                face_path = os.path.join(destination_folder, face_filename)

                # Save the face image
                face_image.save(face_path)
                print(f"Face detected in {filename} and saved as {face_filename}")
            else:
                print(f"No face detected in {filename}")

if __name__ == "__main__":

    # Example usage
    source_folder = "/home/mwaltz/sampleImages"  # Path to the folder containing sampled images
    destination_folder = "/home/mwaltz/sampleImages/faces"  # Path to the folder to save detected faces

    # Load models
    face_model, _ = load_models()

    # Call the function to detect and save faces
    save_detected_faces(source_folder=source_folder, destination_folder=destination_folder, face_model=face_model)
