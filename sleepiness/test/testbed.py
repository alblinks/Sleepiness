import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from warnings import warn

from sleepiness.test import models
from sleepiness.utility.misc import Loader
from sleepiness.end2end.utils import transform as test_transform

# ANSI escape codes for colors and styles
HEADER = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def with_loader(func):
    """
    Decorator to run a function with an 
    animated loader.
    """
    def wrapper(*args, **kwargs):
        if "n_samples" in kwargs:
            n_samples = kwargs["n_samples"]
        with Loader(desc=f"Evaluating model on {n_samples} samples"):
            return func(*args, **kwargs)
    return wrapper

# Helper functions to calculate classification metrics
def recall(outputs: torch.Tensor, 
           labels: torch.Tensor, 
           class_index: int
    ):
    """
    Calculate the recall for a given class.
    """
    true_positives = (outputs == class_index) & (labels == class_index)
    actual_positives = (labels == class_index)
    if actual_positives.sum() == 0:
        warn(f"No actual positives for class {class_index}. Returning 0.0.")
        return 0
    recall = true_positives.sum() / actual_positives.sum()
    return recall

def precision(outputs: torch.Tensor, 
              labels: torch.Tensor, 
              class_index: int
    ):
    """
    Calculate the precision for a given class.
    """
    true_positives = (outputs == class_index) & (labels == class_index)
    predicted_positives = (outputs == class_index)
    if predicted_positives.sum() == 0:
        warn(f"No predicted positives for class {class_index}. Returning 0.0.")
        return 0
    precision = true_positives.sum() / predicted_positives.sum()
    return precision

def f1_score(outputs: torch.Tensor, 
             labels: torch.Tensor, 
             class_index: int
    ):
    """
    Calculate the F1 score for a given class.
    """
    prec = precision(outputs, labels, class_index)
    rec = recall(outputs, labels, class_index)
    f1 = 2 * (prec * rec) / (prec + rec)
    return f1

# Function to create a DataLoader for test images
def create_test_dataloader(test_data_folder: str, batch_size:int = 32):
    """
    Create a DataLoader for the test dataset.
    """
    test_dataset = datasets.ImageFolder(root=test_data_folder, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return test_loader

# Function to evaluate the model using the test DataLoader
@with_loader
def evaluate_model(model: torch.nn.Module, 
                   test_loader: DataLoader, 
                   device: torch.device,
                   n_samples: int = 1000):
    model.eval()  # Set the model to evaluation mode
    model.to(device)
    
    max_v_batch = n_samples // test_loader.batch_size
    
    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients
        cbatch = 0
        for inputs, labels in test_loader:
            if cbatch > max_v_batch:
                break
            inputs: torch.Tensor; labels: torch.Tensor
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            cbatch += 1

    accuracy = accuracy = 100 * correct / total

    # Calculate class-wise precision, recall, and F1 score
    class_names = test_loader.dataset.classes
    
    # Print nice header
    print_colorful_header()
    
    # This enumeration only works because the classes are ordered 
    # alphabetically or in the same order as the labels
    predicted = predicted.to(device)
    labels = labels.to(device)
    for i, class_name in enumerate(class_names):
        rec = recall(predicted, labels, i)
        prec = precision(predicted, labels, i)
        if rec == 0 and prec == 0:
            warn(f"No actual or predicted positives for class {class_name}. "
                 "F1 Score is 0.0.")
            f1 = 0
        f1 = f1_score(predicted, labels, i)
        print_colorful_metrics(class_name, accuracy, prec, rec, f1)

class YourModel(torch.nn.Module):
    """
    This is a placeholder for your PyTorch model. 
    Ensure this class defines your model architecture, and you have a trained model instance.
    """
    def __init__(self):
        super(YourModel, self).__init__()
        # Define your model's architecture here

    def forward(self, x):
        # Define the forward pass here
        return x

def print_colorful_header():
    """
    Prints a colorful and formatted header for the metrics table.
    """
    print(f"{BOLD}{UNDERLINE}{'Class'.center(10)}{ENDC}" +
          f"| {BOLD}{UNDERLINE}{'Accuracy'.center(10)}{ENDC}" +
          f"| {BOLD}{UNDERLINE}{'Precision'.center(10)}{ENDC}" +
          f"| {BOLD}{UNDERLINE}{'Recall'.center(10)}{ENDC}" +
          f"| {BOLD}{UNDERLINE}{'F1 Score'.center(10)}{ENDC}")
    
def print_colorful_metrics(class_name, accuracy, precision, recall, f1):
    """
    Prints a colored and formatted table of Accuracy, Precision, Recall, and F1 Score for each class.
    """
    # Calculate metrics
    col_width = 10

    # Print metrics for each class
    print(f"{BOLD}{class_name.ljust(col_width)}{ENDC}" +
            f"| {FAIL}{accuracy:^{col_width-1}.2f}{ENDC} " +
            f"| {GREEN}{precision:^{col_width-1}.2f}{ENDC} " +
            f"| {BLUE}{recall:^{col_width-1}.2f}{ENDC} " +
            f"| {WARNING}{f1:^{col_width-1}.2f}{ENDC}")

if __name__ == "__main__":
    # Path to your test images (organized in folders by class)
    test_data_folder = 'pictures/e2e_dataset/test'
    
    # Assuming CUDA is available, use GPU for evaluation; otherwise, use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize your model here and load the trained weights
    model = models.SleepinessE2E()

    # Create a DataLoader for your test data
    test_loader = create_test_dataloader(test_data_folder)

    # Evaluate the model
    evaluate_model(model, test_loader, device, n_samples=100)