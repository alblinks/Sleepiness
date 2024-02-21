import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from empty_seat import is_empty

# Load trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp2/weights/last.pt', force_reload=True)

# Classify img
#img = os.path.join('data', 'images', 'awake.8a08edee-ca53-11ee-9dae-6bc3a5752140.jpg')
#results = model(img)

# Display it
#plt.imshow(np.squeeze(results.render()))
#plt.show()
#plt.savefig("Example.pdf")

# Access camera
cap = cv2.VideoCapture("/dev/video0")

while cap.isOpened():

    # Read current frame
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    # Render
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    # Exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

