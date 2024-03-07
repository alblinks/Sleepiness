from torchvision import transforms

IMAGE_WIDTH = 100
IMAGE_HEIGHT = 116

# Define transformations
transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),  # Resize images to 100x116 pixels
    transforms.AutoAugment(),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])