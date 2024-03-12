import cv2
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

N_AUGS_PER_IMG = 3


# Path to the directory containing original images
original_images_dir = "/home/mwaltz/balanced/eyes/closed"

# Path to the directory where augmented images will be saved
augmented_images_dir = "/home/mwaltz/balanced/eyes/closed_aug"

# Create the directory for augmented images if it doesn't exist
if not os.path.exists(augmented_images_dir):
    os.makedirs(augmented_images_dir)

# Initialize the ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Load original images, perform data augmentation, and save both original and augmented images
for i, filename in enumerate(os.listdir(original_images_dir)):

    if i % 100 == 0:
        print(f"{i} images augmented.")

    if filename.endswith(".jpg") or filename.endswith(".png"):

        # Load the original image
        original_image_path = os.path.join(original_images_dir, filename)
        original_image = load_img(original_image_path)
        original_image_array = img_to_array(original_image)

        # Convert BGR to RGB if needed
        if original_image_array.shape[2] == 3:
            original_image_array = original_image_array[..., ::-1]  # Reverse the channel order (BGR to RGB)
        
        # Prep
        original_image_array = original_image_array.reshape((1,) + original_image_array.shape)

        # Save the original image to the augmented images directory
        original_image_save_path = os.path.join(augmented_images_dir, filename)
        original_image.save(original_image_save_path)
        
        # Perform data augmentation
        augmented_images = datagen.flow(original_image_array, batch_size=1)
        
        # Generate and save the augmented images
        for i in range(N_AUGS_PER_IMG):
            augmented_image = augmented_images.next()[0].astype("uint8")

            # Convert BGR to RGB if needed
            #if augmented_image.shape[2] == 3:  # Check if the image has 3 channels
            #    augmented_image = augmented_image[..., ::-1]  # Reverse the channel order (BGR to RGB)
            
            augmented_image_filename = os.path.splitext(filename)[0] + f"_aug_{i}.jpg"  # Adjust the filename format as needed
            augmented_image_path = os.path.join(augmented_images_dir, augmented_image_filename)
            cv2.imwrite(augmented_image_path, augmented_image)
