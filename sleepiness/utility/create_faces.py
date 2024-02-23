import os
from PIL import Image
from ultralytics import YOLO
from sleepiness.face.detectFace import load_face_model, detect_face


def save_detected_faces(source_folder : str, destination_folder : str, face_model : YOLO) -> None:
    """Detect and save faces in new images."""

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Loop over the images in the source folder
    for i, filename in enumerate(os.listdir(source_folder)):
        
        if i % 100 == 0:
            print(f"{i} images processed.")

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

if __name__ == "__main__":

    # Example usage
    source_folder = "/home/mwaltz/sampleImages/raw"
    destination_folder = "/home/mwaltz/sampleImages/faces"

    # Load model
    face_model = load_face_model()

    # Call the function to detect and save faces
    save_detected_faces(source_folder=source_folder, destination_folder=destination_folder, face_model=face_model)
