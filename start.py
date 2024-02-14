import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

# Load model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Classify example img
#img = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Cars_in_traffic_in_Auckland%2C_New_Zealand_-_copyright-free_photo_released_to_public_domain.jpg/800px-Cars_in_traffic_in_Auckland%2C_New_Zealand_-_copyright-free_photo_released_to_public_domain.jpg'
#results = model(img)
#results.print()
#plt.imshow(np.squeeze(results.render()))
#plt.savefig("Cars.pdf")

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

