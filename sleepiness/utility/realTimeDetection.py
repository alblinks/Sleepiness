from collections import deque
import cv2
import sleepiness.pipelines as pipelines
from sleepiness.test.aggregators import LabelAggregator, MajorityVoting

def framewise_real_time_detection(model: pipelines.Pipeline) -> None:
    """
    Real time classification for every frame from the camera.
    """
    # Access camera
    cap = cv2.VideoCapture("/dev/video0")

    while cap.isOpened():

        # Read current frame
        ret, frame = cap.read()
        assert ret, "Error reading frame"
        # Make detections 
        result = model.classify(frame)
        
        # Write a text in the top right corner
        cv2.putText(frame, f"{result}", (10, 10), cv2.QT_FONT_NORMAL, 0.5, (0, 255, 0), 2)
        # Render
        cv2.imshow(f"Real-Time Detection", frame)
        
        # Exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

def aggregated_real_time_detection(model: pipelines.Pipeline, 
                                   aggregator: LabelAggregator) -> None:
    """
    Real time classification for every frame from the camera.
    """
    # Access camera
    cap = cv2.VideoCapture("/dev/video0")
    while cap.isOpened():

        # Read current frame
        ret, frame = cap.read()
        assert ret, "Error reading frame"
        # Make detections 
        result = model.classify(frame)
        aggregator.add(result)
        
        # Write the aggregated result in the top right corner
        cv2.putText(frame, f"{aggregator.state}", 
                    (10, 10), cv2.QT_FONT_NORMAL, 
                    0.5, (150, 25, 160), 1
                )

        # Write the individual results in the bottom right corner
        for i, state in enumerate(aggregator.labels):
            cv2.putText(frame, f"{state}", 
                        (10, 10 + 20 * (i + 1)), cv2.QT_FONT_NORMAL, 
                        0.5, (0, 180, 60), 1)

        # Render
        cv2.imshow(f"Real-Time Detection", frame)
        
        # Exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

if __name__ == "__main__":
    model = pipelines.FullPipeline(
        eye_model_confidence=0.2 ,
        hand_model_confidence=0.5,
    )
    #framewise_real_time_detection(model)
    aggregator = MajorityVoting()
    aggregated_real_time_detection(model, aggregator)
