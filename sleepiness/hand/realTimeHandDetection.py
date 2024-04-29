import cv2
from sleepiness.hand.detection import detect_from_numpy_array, draw_landmarks_on_image


if __name__ == "__main__":

    # Access camera
    cap = cv2.VideoCapture("/dev/video0")

    while cap.isOpened():

        # Read current frame
        ret, frame = cap.read()

        # Inference 
        detection_result = detect_from_numpy_array(frame)
        if detection_result:
            annotated_image = draw_landmarks_on_image(frame, detection_result)
            frame = annotated_image
        
        # Display the image
        cv2.imshow("Hand Detection", frame)

        # Exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
