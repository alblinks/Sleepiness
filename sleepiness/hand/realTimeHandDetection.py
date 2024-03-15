import cv2
from sleepiness.hand import load_model


if __name__ == "__main__":

    # Load hand model
    hand_model = load_model()

    # Access camera
    cap = cv2.VideoCapture("/dev/video0")

    while cap.isOpened():

        # Read current frame
        ret, frame = cap.read()

        # Inference 
        width, height, inference_time, results = hand_model.inference(frame)

        # Display hands
        for detection in results:
            id, name, confidence, x, y, w, h = detection

            # draw a bounding box rectangle and label on the image
            color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
        
        # Display the image
        cv2.imshow("Hand Detection", frame)

        # Exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
