import os
import shutil

# Define input and output directories
input_dir = "/home/mwaltz/train"
output_dir = "/home/mwaltz/train_sleeping"

# Loop through subfolders in the input directory
for root, dirs, files in os.walk(input_dir):
    for dir_name in dirs:
        # Check if the current subfolder contains a subfolder named "sleeping"
        sleeping_dir = os.path.join(root, dir_name, "sleeping")

        print(sleeping_dir)
        if os.path.exists(sleeping_dir) and os.path.isdir(sleeping_dir):
            # Create corresponding "sleeping" folder in the output directory
            output_sleeping_dir = os.path.join(output_dir, dir_name)
            os.makedirs(output_sleeping_dir, exist_ok=True)
            
            # Copy the contents of the "sleeping" folder to the output directory
            for item in os.listdir(sleeping_dir):
                item_path = os.path.join(sleeping_dir, item)
                if os.path.isfile(item_path):
                    shutil.copy2(item_path, output_sleeping_dir)

print("Done copying 'sleeping' folder contents.")
