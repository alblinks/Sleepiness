import uuid
import os
import time
import cv2

IMAGES_PATH = os.path.join("data", "images") #/data/images
labels = ["sleepy"] # awake, sleepy
number_imgs = 10

# Access camera
cap = cv2.VideoCapture("/dev/video0")

# Loop through labels
for label in labels:
    print(f"Collecting images for {label}")
    time.sleep(5)
    
    # Loop through image range
    for img_num in range(number_imgs):
        print(f"Collecting images for {label}, image number {img_num}")
        
        # Webcam feed
        ret, frame = cap.read()
        
        # Naming out image path
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        
        # Writes out image to file 
        cv2.imwrite(imgname, frame)
        
        # Render to the screen
        #cv2.imshow('Image Collection', frame)
        
        # 2 second delay between captures
        time.sleep(2)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
