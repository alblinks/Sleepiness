import shutil
import os
import uuid

# Source folder path
# subject_0001_bright, subject_0003_dimmed, subject_0004_dimmed, subject_0084_bright, subject_0087_bright
# subject_0054_bright
source_folder = '/home/mwaltz/balanced/labeled/subject_0054_bright/awake'

# Destination folder path
destination_folder = '/home/mwaltz/balanced/clearly_awake'

uid = str(uuid.uuid4())

# Iterate through files in the source folder
for filename in os.listdir(source_folder):
    if filename.startswith('fram') and filename.endswith('.jpg'):
        # Extract the number from the filename
        file_number = int(filename[4:-4])  # Assuming the format is 'fram<number>.jpg'
        
        # Check if the number is less than or equal to 3000
        if file_number <= 500:
            # Construct source and destination paths
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, uid + filename)
            
            # Copy the file to the destination folder
            shutil.copy(source_path, destination_path)
            print(f"Copied {filename} to {destination_folder}")
