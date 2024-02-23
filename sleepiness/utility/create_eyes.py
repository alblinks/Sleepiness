import os
import cv2
import supervision as spv
from sleepiness.eye.detectEye import load_eye_model


# Function to detect eyes in an image and save if detected
def detect_and_save_eyes(image_path, eye_model, output_dir):

    print(f"Processing image {image_path}")

    # Read the image
    frame = cv2.imread(image_path)

    # Inference
    result = eye_model(frame, agnostic_nms=True, verbose=False)[0]
    detections = spv.Detections.from_yolov8(result)

    # Loop through detections and save the regions of detected eyes
    eye_cnt = 0

    for i, detection in enumerate(detections):

        # Class index of eyes is 0
        if detection[2] == 0:
            x_min, y_min, x_max, y_max = detection[0]
            eye_region = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            eye_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_eye_{eye_cnt}.jpg"
            cv2.imwrite(os.path.join(output_dir, eye_filename), eye_region)

            eye_cnt += 1

if __name__ == "__main__":

    # Directory containing images
    input_dir = "/home/mwaltz/sampleImages/raw"

    # Directory to save images with detected eyes
    output_dir = "/home/mwaltz/sampleImages/eyes"

    # Load eye model
    eye_model = load_eye_model()

    # Loop over images in the input directory
    for i, filename in enumerate(os.listdir(input_dir)):
        if i % 100 == 0:
            print(f"{i} images processed.")

        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            detect_and_save_eyes(image_path, eye_model, output_dir)
