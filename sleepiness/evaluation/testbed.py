from typing import Callable
from torchvision import datasets
from torch.utils.data import DataLoader

from sleepiness.evaluation.utils import TimeSeriesDataLoader, logger


# Function to create a DataLoader for test images
def create_test_dataloader(test_data_folder: str, 
                           batch_size:int = 32,
                           transform: Callable | None = None):
    """
    Create a DataLoader for the test dataset.
    """
    try:
        test_dataset = datasets.ImageFolder(root=test_data_folder, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return test_loader
    except FileNotFoundError:
        logger.warning(
            f"Torchvision DataLoader failed to load data from {test_data_folder}.\n"
            "Using TimeSeriesDataLoader as a fallback.")
        return TimeSeriesDataLoader(test_data_folder)