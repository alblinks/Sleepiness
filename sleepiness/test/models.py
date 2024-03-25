from __future__ import annotations
from typing import Callable
import numpy as np

import torch
from PIL import Image
from abc import abstractmethod
from functools import partial
from random import shuffle

from torch import Tensor
from torchvision import models
from torch.utils.data import DataLoader

# Import trained weight paths
from sleepiness.end2end.weights import __path__ as e2eWeights
from sleepiness.empty_seat.CNN.weights import __path__ as emptyWeightsCNN
from sleepiness.empty_seat.FFNN.weights import __path__ as emptyWeightsFFNN

# Load hand, eye and face modules
import sleepiness.hand as hand
import sleepiness.eye as eye
import sleepiness.face as face

# Import PassengerState
from sleepiness.utility.pstate import (
    reduce_state, ReducedPassengerState,
    PassengerState
)

# import full pipeline
import sleepiness.pipelines as pipelines

from sleepiness.empty_seat.pixdiff import (
    __path__ as pixdiffPath , preprocess
)
import sleepiness.test.utils as tutils
from sleepiness.utility.logger import logger

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
    
    @abstractmethod
    def print_scores(self,
                     all_predictions: Tensor, 
                     all_labels: Tensor,
                     class_names: list[str]):
        """
        Print the accuracy, precision, recall, and F1 score.
        """
        all_predictions = torch.tensor(all_predictions)
        all_labels = torch.tensor(all_labels)
        tutils.MetricsCollector(
            self, all_predictions, all_labels, class_names
        ).summary()
    
    @tutils.with_loader
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

        # Calculate class-wise precision, recall, and F1 score
        class_names = test_loader.dataset.classes

        self.print_scores(all_predictions, all_labels, class_names)

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
        self.pixmap: np.ndarray = pickle.load(open(f"{pixdiffPath[0]}/avgmap.nparray", "rb"))
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
            
    @tutils.with_loader
    def evaluate(self,
                test_loader: DataLoader,
                device: torch.device,
                n_samples: int = 1000):
        """
        Evaluate the model on the test dataset.
        """
        logger.info(
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
                
        # Calculate class-wise precision, recall, and F1 score
        class_names = test_loader.dataset.classes
        
        self.print_scores(all_predictions, all_labels, class_names)
                    
class ReducedFullPipeline(EvalClassifier):
    """
    Reduced pipeline model:
    
    Here, we only classify AWAKE and SLEEPING states.
    NOTTHERE is not considered.
    """
    def __init__(self, 
                 eye_model_confidence: float = 0.2,
                 hand_model_confidence: float = 0.5):
        
        self.eye_model_confidence = eye_model_confidence
        self.hand_model_confidence = hand_model_confidence
        super().__init__()
    
    def load_model(self):
        """
        Load the pre-trained model.
        """
        pipeline = pipelines.FullPipeline(
            eye_model_confidence=self.eye_model_confidence,
            hand_model_confidence=self.hand_model_confidence
        )
        self.model = pipeline.classify
        
        return self
        
    def forward(self, x: Tensor) -> Tensor:
        return NotImplementedError(
            "The full pipeline model does not "
            "have a single forward pass."
        )
    
    @tutils.with_loader
    def evaluate(self,
                test_loader: DataLoader,
                device: torch.device,
                n_samples: int = 1000):
        """
        Evaluate the model on the test dataset.
        """
        logger.warning(
            "This model has multiple nets. "
            "`device` will be ignored."
        )
        logger.warning(
            "Model uses raw pictures. All transforms "
            "given by the test loader are ignored."
        )
        del device
        
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        cbatch = 0
        imgs = test_loader.dataset.imgs.copy()
        shuffle(imgs)
        for idx, (path_to_img, label) in enumerate(imgs):
            if idx == n_samples:
                break
                
            # Classify the image
            state = self.model(path_to_img=path_to_img)
            
            # Coerce to ReducedPassengerState
            state = reduce_state(state)
            if state is ReducedPassengerState.NOTAVAILABLE:
                logger.warning(
                    f"Image {path_to_img} is classified as NOT_THERE. "
                    "Skipping in reduced pipeline."
                )
                continue
            
            # Classify state
            if state is ReducedPassengerState.AWAKE:
                prediction = ReducedPassengerState.AWAKE.value
                all_predictions.append(prediction)
            elif state is ReducedPassengerState.SLEEPING:
                prediction = ReducedPassengerState.SLEEPING.value
                all_predictions.append(prediction)
            all_labels.append(label)
            total += 1
            correct += int(prediction == label)
            cbatch += 1
                
        # Calculate class-wise precision, recall, and F1 score
        class_names = test_loader.dataset.classes

        self.print_scores(all_predictions, all_labels, class_names)

class FullPipeline(ReducedFullPipeline):
    """
    Full pipeline model:
    
    Here, we classify all three states:
    AWAKE, SLEEPING, and NOT_THERE.
    """
    
    @tutils.with_loader
    def evaluate(self,
                test_loader: DataLoader,
                device: torch.device,
                n_samples: int = 1000):
        """
        Evaluate the model on the test dataset.
        """
        logger.info(
            "This model has multiple nets. "
            "`device` control is taken care of."
        )
        logger.warning(
            "Model uses raw pictures. All transforms "
            "given by the test loader are ignored."
        )
        del device
        
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        cbatch = 0
        imgs = test_loader.dataset.imgs.copy()
        shuffle(imgs)
        for idx, (path_to_img, label) in enumerate(imgs):
            if idx == n_samples:
                break
                
            # Classify the image
            state: PassengerState = self.model(path_to_img=path_to_img)
            
            prediction = state.value
            all_predictions.append(prediction)
            all_labels.append(label)
            total += 1
            correct += int(prediction == label)
            cbatch += 1
                
        # Calculate class-wise precision, recall, and F1 score
        class_names = test_loader.dataset.classes
    
        self.print_scores(all_predictions, all_labels, class_names)
        
class NoEyesReducedPipeline(ReducedFullPipeline):
    """
    Reduced pipeline model:
    
    Here, we only classify AWAKE and SLEEPING states.
    NOTTHERE is not considered.
    """
    def __init__(self):
        pass

    
    def load_model(self):
        """
        Load the pre-trained model.
        """
        pipeline = pipelines.NoEyePipeline()
        self.model = pipeline.classify
        
        return self