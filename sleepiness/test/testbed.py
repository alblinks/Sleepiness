from typing import Callable
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision

from sleepiness.test import models
from sleepiness.test.utils import *

# These imports are necessary to load the models
# via pickle.
from sleepiness.empty_seat.CNN.train import Emptyfier as Emtpyfier # typo in the class name
from sleepiness.end2end.utils import transform as e2e_transform
from sleepiness.empty_seat.CNN.utils import transform as empty_transform
from sleepiness.eye import CustomCNN

# Function to create a DataLoader for test images
def create_test_dataloader(test_data_folder: str, 
                           batch_size:int = 32,
                           transform: Callable | None = None):
    """
    Create a DataLoader for the test dataset.
    """
    test_dataset = datasets.ImageFolder(root=test_data_folder, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return test_loader

if __name__ == "__main__":
    # Path to your test images (organized in folders by class)
    test_data_folder = "pictures/balanced_correct_full/test"
    # test_data_folder = "pictures/e2e_dataset/test"
    
    # Assuming CUDA is available, use GPU for evaluation; otherwise, use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.NoEyesReducedPipeline().load_model()

    # Create a DataLoader for your test data
    test_loader = create_test_dataloader(
        test_data_folder, 
        batch_size=32, 
        transform=torchvision.transforms.ToTensor()
    )

    # Evaluate the model
    model.evaluate(test_loader, device, n_samples=100)