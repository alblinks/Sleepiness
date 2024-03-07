from __future__ import annotations
import numpy as np

import torch
from PIL import Image
from abc import abstractmethod

from warnings import warn
from torch import Tensor
from torchvision import models
from torch.utils.data import DataLoader
from sleepiness.end2end.weights import __path__ as e2eWeights
from sleepiness.empty_seat.CNN.weights import __path__ as emptyWeightsCNN
from sleepiness.empty_seat.FFNN.weights import __path__ as emptyWeightsFFNN
from sleepiness.empty_seat.pixdiff import (
    __path__ as pixdiffPath , preprocess
)
from sleepiness.test.utils import (
    with_loader, ClassifierMetrics,
    ClassifierMetricsPrinter
)

class EvalClassifier(torch.nn.Module):
    """
    Evaluation class for trained classifiers.
    
    This class is an abstract class that defines the
    interface for evaluating a trained classifier.
    
    The class provides a method to evaluate the model
    on a test dataset and calculate the accuracy, precision,
    recall, and F1 score.
    
    For neural network-based models, the class provides
    a pre-implementation of the `evaluate` method that
    takes care of the forward pass through the model.
    
    For non-neural network-based models, you need to
    manually override the `evaluate` method to implement
    the evaluation logic.
    """
    def __init__(self):
        super().__init__()
        self.model: torch.nn.Module
        return None
    
    @abstractmethod
    def load_model(self) -> EvalClassifier:
        """
        Load the pre-trained model as self.model.
        """
        raise NotImplemented
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model.
        """
        raise NotImplementedError
    
    @with_loader
    def evaluate(self,
                test_loader: DataLoader, 
                device: torch.device,
                n_samples: int = 1000):
        """
        Evaluate the model on the test dataset.
        """
        
        self.model.eval()  # Set the model to evaluation mode
        self.model.to(device)
        
        max_v_batch = n_samples // test_loader.batch_size
        
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():  # No need to track gradients
            cbatch = 0
            for inputs, labels in test_loader:
                if cbatch > max_v_batch:
                    break
                inputs: torch.Tensor; labels: torch.Tensor
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted)
                all_labels.extend(labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                cbatch += 1

        accuracy = 100 * correct / total

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

class SleepinessE2E(EvalClassifier):
    """
    Fine-tuned ResNet50 for End-2-End sleepiness detection.
    """
    def __init__(self):
        super().__init__()
        return None
    
    def load_model(self):
        """
        Load the pre-trained model.
        """
        super(SleepinessE2E, self).__init__()
        # Load the pre-trained model
        self.model: models.ResNet = torch.load(f"{e2eWeights[0]}/ResNet50_e2e_4.pt")
        
        # Freeze the pre-trained layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Replace the last fully connected layer with a new one
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 2),
            torch.nn.LogSoftmax(dim=1)
        )
        return self
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

class EmptyfierCNN(EvalClassifier):
    """
    Class for the empty seat detection model.
    """
    def __init__(self):
        super().__init__()
    
    def load_model(self):
        """
        Load the pre-trained model.
        """
        self.model: torch.nn.Module = torch.load(f"{emptyWeightsCNN[0]}/empty_seat_epoch_5.pt")
        return self
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
class EmptyfierFFNN(EvalClassifier):
    """
    Class for the empty seat detection model.
    """
    def __init__(self):
        super().__init__()
    
    def load_model(self):
        """
        Load the pre-trained model.
        """
        self.model: torch.nn.Module = torch.load(f"{emptyWeightsFFNN[0]}/empty_seat_epoch_5.pt")
        return self
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
    
class EmptyfierPixDiff(EvalClassifier):
    """
    Class for the empty seat detection model
    using average pixel difference.
    """
    # Average normalized pixel difference threshold
    # to classify an image as empty or not.
    # Images with pixel difference greater than this
    # threshold are classified as occupied.
    THRESHOLD = 0.08
    
    def __init__(self):
        super().__init__()
        # Load the pre-trained model
    
    def load_model(self):
        """
        Load the pre-trained model.
        """
        import pickle
        self.pixmap: np.ndarray = pickle.load(open(f"{pixdiffPath[0]}/avgmap.pkl", "rb"))
        return self
        
    def forward(self, x: Tensor) -> Tensor:
        return NotImplementedError(
            "This model does not have a forward pass."
        )
    
    def pixdiff(self, image: np.ndarray) -> float:
        """
        Calculate the pixel difference between an image
        and the AVGMAP.
        """
        return np.abs(image - self.pixmap).mean()
            
    @with_loader
    def evaluate(self,
                test_loader: DataLoader,
                device: torch.device,
                n_samples: int = 1000):
        """
        Evaluate the model on the test dataset.
        """
        print(
            "This model is not neural network-based. "
            "`device` settings will be ignored."
        )
        del device
        
        max_v_batch = n_samples // test_loader.batch_size
        
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        cbatch = 0
        for inputs, labels in test_loader:
            if cbatch > max_v_batch:
                break
            inputs: torch.Tensor; labels: torch.Tensor
            inputs, labels = inputs.numpy(), labels.numpy()
            
            for input, label in zip(inputs, labels):
                
                # Rearrange dims to (H, W, C)
                input = np.moveaxis(input, 0, -1)
                
                # Rescale to 0-255
                input = (input * 255).astype(np.uint8)
                
                # Transform the input to a PIL image
                # THERE / AWAKE = 0, NOT_THERE = 1
                input = Image.fromarray(input)
                input = preprocess(input)
                
                if self.pixdiff(input) > self.THRESHOLD:
                    prediction = 0
                    all_predictions.append(prediction)
                else:
                    prediction = 1
                    all_predictions.append(prediction)
                all_labels.append(label)
                total += 1
                correct += int(prediction == label)
                cbatch += 1
                
        accuracy = 100 * correct / total
        
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