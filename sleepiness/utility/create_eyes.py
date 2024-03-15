import os
import cv2
import supervision as spv
import random
import sleepiness.face as face
import sleepiness.eye as eye

def detect_and_save_eyes(image_path, face_model, eye_model, output_dir):
    """Function to detect eyes in an image and save if detected."""

    # Read the image
    img = cv2.imread(image_path)

    # Detect largest face
    #face_detected, faceImg = detect_face(img=img, face_model=face_model, with_xyxy=False)
    face_detected = True

    if face_detected:

        # Detect eyes
        eye_regions, _ = eye.detect(faceImg=img, eye_model=eye_model)

        # Save them
        for i, r in enumerate(eye_regions):
            
            # Resize the region
            r = cv2.resize(r, (50, 20)) # width, height

            # Store
            eye_filename = os.path.splitext(os.path.basename(image_path))[0] + f"-eye-{i}.jpg"
            cv2.imwrite(os.path.join(output_dir, eye_filename), r)

if __name__ == "__main__":

    # Directory containing images
    input_dir = "/home/mwaltz/balanced/train/sleeping"

    # Directory to save images with detected eyes
    output_dir = "/home/mwaltz/balanced/eyes/closed"

    # Load models
    face_model = face.load_model()
    eye_model  = eye.load_model()

    # Loop over images in the input directory
    x = os.listdir(input_dir)
    random.shuffle(x)
    N = len(x)

    for i, filename in enumerate(x):

        if i >= 1000:
            break

        if i % 10 == 0:
            print(f"{i} of {N} images processed.")

        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            try:
                detect_and_save_eyes(image_path=image_path, face_model=face_model, 
                                    eye_model=eye_model, output_dir=output_dir)
            except:
                print(f"Error with image {filename}. Skipping..")
