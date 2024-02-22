import cv2
from detectFace import load_face_model


if __name__ == "__main__":

    # Load trained model
    face_model = load_face_model()
    
    # Access camera
    cap = cv2.VideoCapture("/dev/video0")

    while cap.isOpened():

        # Read current frame
        ret, frame = cap.read()
        
        # Make detections using YOLO
        results = face_model.predict(frame, stream=False)
        
        # Render
        # View results
        #for r in results:
        #    print(r.boxes)  
        #rendered_frame = np.squeeze(results.render())
        #cv2.imshow('YOLO', rendered_frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("Face Detection", annotated_frame)
        
        # Exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
