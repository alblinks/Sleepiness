import cv2
from sleepiness.pipelines import NoEyePipeline

# Load trained model
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp2/weights/last.pt', force_reload=True)

# Classify img
#img = os.path.join('data', 'images', 'awake.8a08edee-ca53-11ee-9dae-6bc3a5752140.jpg')
#results = model(img)

# Display it
#plt.imshow(np.squeeze(results.render()))
#plt.show()
#plt.savefig("Example.pdf")

model = NoEyePipeline()

# Access camera
cap = cv2.VideoCapture("/dev/video0")

while cap.isOpened():

    # Read current frame
    ret, frame = cap.read()
    # Make detections 
    result = model.classify(frame)
    
    # Write a text in the top right corner
    cv2.putText(frame, f"{result}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Render
    cv2.imshow(f"Detection", frame)
    
    
    # Exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

