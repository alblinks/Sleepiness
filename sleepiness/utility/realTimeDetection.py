from collections import deque
import cv2
import os
import sleepiness.pipelines as pipelines
from sleepiness.evaluation.aggregators import LabelAggregator, MajorityVoting


def framewise_real_time_detection(model: pipelines.FullPipeline, draw_bbox: bool = False) -> None:
    """
    Real time classification for every frame from the camera.
    """
    # Access camera
    # Check if on Windows
    if os.name == 'nt':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture("/dev/video0")
    
    crop_factors = model.hand_model_crop

    while cap.isOpened():

        # Read current frame
        ret, frame = cap.read()
        assert ret, "Error reading frame"
        # Make detections
        result, bboxes = model.classify(frame, return_bbox=True)

        if draw_bbox:
            face_xxyy, eyes_xxyy, hands_xxyy = bboxes
            cropped_img = pipelines.crop_image(frame, *crop_factors)
            # Draw face bounding box
            if face_xxyy is not None:
                cv2.rectangle(
                    frame, (face_xxyy[0], face_xxyy[2]), (face_xxyy[1], face_xxyy[3]), (0, 255, 0), 2)

            # Draw bounding boxes for eyes
            for eye_xxyy in eyes_xxyy:

                # Consider the eye coordinates are for face img
                xmin = eye_xxyy[0] + face_xxyy[0]
                xmax = eye_xxyy[1] + face_xxyy[0]
                ymin = eye_xxyy[2] + face_xxyy[2]
                ymax = eye_xxyy[3] + face_xxyy[2]
                cv2.rectangle(frame, (xmin, ymin),
                              (xmax, ymax), (0, 0, 255), 2)

            # Draw bounding boxes for hands
            for hand_xxyy in hands_xxyy:
                hand_xxyy = model.transform_xxyy_for_cropped_img(
                    full_img=frame, cropped_img=cropped_img, xxyy=hand_xxyy
                )
                cv2.rectangle(
                    frame,
                    (hand_xxyy[0], hand_xxyy[2]),
                    (hand_xxyy[1], hand_xxyy[3]),
                    (255, 0, 0),
                    2
                )

        # Write a text in the top right corner
        cv2.putText(frame, f"{result}", (10, 10),
                    cv2.QT_FONT_NORMAL, 0.5, (0, 255, 0), 2)
        # Render
        cv2.imshow(f"Real-Time Detection", frame)

        # Exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return


def aggregated_real_time_detection(model: pipelines.FullPipeline,
                                   aggregator: LabelAggregator,
                                   draw_bbox: bool = False) -> None:
    """
    Real time classification for every frame from the camera.
    """
    crop_factors = model.hand_model_crop

    # Access camera
    if os.name == 'nt':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture("/dev/video0")

    while cap.isOpened():

        # Read current frame
        ret, frame = cap.read()
        assert ret, "Error reading frame"
        # Make detections
        result, bboxes = model.classify(frame, return_bbox=True)
        aggregator.add(result)
        
        if draw_bbox:
            face_xxyy, eyes_xxyy, hands_xxyy = bboxes
            cropped_img = pipelines.crop_image(frame, *crop_factors)
            # Draw face bounding box
            if face_xxyy is not None:
                cv2.rectangle(
                    frame, (face_xxyy[0], face_xxyy[2]), (face_xxyy[1], face_xxyy[3]), (0, 255, 0), 2)

            # Draw bounding boxes for eyes
            for eye_xxyy in eyes_xxyy:

                # Consider the eye coordinates are for face img
                xmin = eye_xxyy[0] + face_xxyy[0]
                xmax = eye_xxyy[1] + face_xxyy[0]
                ymin = eye_xxyy[2] + face_xxyy[2]
                ymax = eye_xxyy[3] + face_xxyy[2]
                cv2.rectangle(frame, (xmin, ymin),
                              (xmax, ymax), (0, 0, 255), 2)

            # Draw bounding boxes for hands
            for hand_xxyy in hands_xxyy:
                hand_xxyy = model.transform_xxyy_for_cropped_img(
                    full_img=frame, cropped_img=cropped_img, xxyy=hand_xxyy
                )
                cv2.rectangle(
                    frame,
                    (hand_xxyy[0], hand_xxyy[2]),
                    (hand_xxyy[1], hand_xxyy[3]),
                    (255, 0, 0),
                    2
                )

        # Write the aggregated result in the top right corner
        cv2.putText(frame, f"{aggregator.state}",
                    (10, 10), cv2.QT_FONT_NORMAL,
                    0.5, (150, 250, 160), 1
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
        eye_model_confidence=0.2,
        hand_model_confidence=0.15,
        hand_model_crop=[0, 1, 0, 1]
    )
    # framewise_real_time_detection(model, draw_bbox=True)
    aggregator = MajorityVoting(horizon=10)
    aggregated_real_time_detection(model, aggregator, draw_bbox=True)
