import cv2
import supervision as spv

from detectEye import load_eye_model

if __name__ == "__main__":

    # Loading the model
    eye_model = load_eye_model()

    # This will draw the detections
    class_colors = spv.ColorPalette.from_hex(['#ffff66', '#66ffcc', '#ff99ff', '#ffcc99'])
    box_annotator = spv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
        color=class_colors
    )

    # Reading frames from the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Inference
        result = eye_model(frame, agnostic_nms=True, verbose=False)[0]
        detections = spv.Detections.from_yolov8(result)

        # Keep only those detections associated with eyes
        eye_detections = []

        for detection in detections:

            # Class index of eyes is 0
            if detection[2] == 0:
                eye_detections.append(detection)

        labels = [
            f"open"
            for _, confidence, _, _
            in eye_detections
        ]
        frame = box_annotator.annotate(
            scene=frame,
            detections=eye_detections,
            labels=labels
        )

        # Display the image
        cv2.imshow("Eye detections", frame)

        # Exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
