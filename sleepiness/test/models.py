import torch

from torch import Tensor
from torchvision import models
from sleepiness.end2end.weights import __path__ as e2eWeights

class SleepinessE2E(torch.nn.Module):
    """
    Fine-tuned ResNet152 for sleepiness detection.
    """
    def __init__(self):
        super(SleepinessE2E, self).__init__()
        # Load the pre-trained model
        self.model: models.ResNet = torch.load(f"{e2eWeights[0]}/ResNet152_e2e.pt")
        
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
        
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)