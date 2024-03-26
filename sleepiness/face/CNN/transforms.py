from torchvision import transforms as _transforms
from torch import Tensor

# Data transformation
train_transform: Tensor = _transforms.Compose([
    _transforms.Resize((200,100)),
    _transforms.RandomHorizontalFlip(),
    _transforms.RandomVerticalFlip(),
    _transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    _transforms.ToTensor(),
])

val_transform: Tensor = _transforms.Compose([
    _transforms.Resize((200,100)),
    _transforms.ToTensor(),
])