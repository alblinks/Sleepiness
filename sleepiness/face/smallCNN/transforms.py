import torch

from torchvision import transforms as _transforms
from torch import Tensor
from PIL import Image


class RescaleIntensityTransform:
    def __call__(self, img):
        # Convert PIL image to PyTorch tensor
        img_tensor = _transforms.ToTensor()(img)
        
        # Rescale the tensor values to [0, 1]
        min_val = torch.min(img_tensor)
        max_val = torch.max(img_tensor)
        img_tensor = (img_tensor - min_val) / (max_val - min_val)
        
        # Scale back to [0, 255]
        img_tensor = img_tensor * 255
        
        # Convert back to PIL image
        img_rescaled = _transforms.ToPILImage()(img_tensor)
        
        return img_rescaled


# Data transformation
train_transform: Tensor = _transforms.Compose([
    _transforms.Resize((200,100)),
    RescaleIntensityTransform(),
    _transforms.RandomHorizontalFlip(),
    _transforms.RandomVerticalFlip(),
    _transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    _transforms.ToTensor(),
])

val_transform: Tensor = _transforms.Compose([
    _transforms.Resize((200,100)),
    RescaleIntensityTransform(),
    _transforms.ToTensor(),
])