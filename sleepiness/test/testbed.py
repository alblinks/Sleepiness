from typing import Callable
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from warnings import warn
import numpy as np
import torchvision

from sleepiness.test import models
from sleepiness.test.utils import *

from sleepiness.empty_seat.CNN.train import Emptyfier as Emtpyfier # typo in the class name
from sleepiness.end2end.utils import transform as e2e_transform
from sleepiness.empty_seat.CNN.utils import transform as empty_transform


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

@with_loader
def evaluate_model_single(model: Callable,
                          n_samples: int = 1000):
    
    if test_loader.batch_size != 1:
        raise ValueError("The DataLoader must have a batch size of 1.")
    
    predictions = []
    ground_truth = []
    with torch.no_grad():  # No need to track gradients
        iterations = 0
        for path, label in test_loader.dataset.imgs:
            if iterations == n_samples:
                break

            prediction = model(path)
            predictions.append(prediction)
            ground_truth.append(label.item())
            
            iterations += 1

    predictions = torch.tensor(predictions)
    ground_truth = torch.tensor(ground_truth)

    # Calculate accuracy
    accuracy = (predictions == ground_truth).sum() / len(ground_truth)

    # Calculate class-wise precision, recall, and F1 score
    class_names = test_loader.dataset.classes
        
    all_predictions = torch.tensor(all_predictions)
    all_labels = torch.tensor(all_labels)
    with ClassifierMetricsPrinter() as printer:
        for lbl_idx, class_name in enumerate(class_names):
            with ClassifierMetrics(all_predictions, all_labels, lbl_idx) as metrics:
                rec = metrics.recall()
                prec = metrics.precision()
                f1 = metrics.f1_score()
                printer.log_metics(class_name,accuracy,prec,rec,f1)

    
if __name__ == "__main__":
    # Path to your test images (organized in folders by class)
    test_data_folder = 'pictures/empty_seat_dataset/test'
    
    # Assuming CUDA is available, use GPU for evaluation; otherwise, use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # To evaluate in batches your model must be a torch.nn.Module.
    # For single image evaluation, you can use a function, 
    # taking the image path as input and returning the class
    # labels as an int.
    # model = models.EmptyfierCNN().load_model()
    #model = lambda x: 0  # Placeholder for your model
    model = models.EmptyfierPixDiff().load_model()

    # Create a DataLoader for your test data
    test_loader = create_test_dataloader(
        test_data_folder, 
        batch_size=32, 
        transform=torchvision.transforms.ToTensor()
    )

    # Evaluate the model
    # evaluate_model_single(model, test_loader, device, n_samples=1000)
    model.evaluate(test_loader, device, n_samples=5000)