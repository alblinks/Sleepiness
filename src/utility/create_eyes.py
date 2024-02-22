import os
import cv2
import supervision as spv


# Function to detect eyes in an image and save if detected
def detect_and_save_eyes(image_path, eye_model, output_dir):
    # Read the image
    frame = cv2.imread(image_path)

    # Inference
    result = eye_model(frame, agnostic_nms=True, verbose=False)[0]
    detections = spv.Detections.from_yolov8(result)

    # Loop through detections and save the regions of detected eyes
    for i, detection in enumerate(detections):

        # Class index of eyes is 0
        if detection[2] == 0:
            x, y, w, h = detection[:4]
            eye_region = frame[int(y):int(y+h), int(x):int(x+w)]
            eye_filename = os.path.splitext(os.path.basename(image_path))[0] + f"_eye_{i}.jpg"
            cv2.imwrite(os.path.join(output_dir, eye_filename), eye_region)

if __name__ == "__main__":

    # Directory containing images
    input_dir = "/home/mwaltz/sampleImages"

    # Directory to save images with detected eyes
    output_dir = "/home/mwaltz/sampleImages/eyes"

    # Load your YOLOv8 eye detection model
    eye_model = load_eye_model()

    # Loop over images in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            detect_and_save_eyes(image_path, eye_model, output_dir)
