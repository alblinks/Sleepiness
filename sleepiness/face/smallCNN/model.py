from torch import nn
from torch.nn import functional as F

# Set up custom CNN Network
class smallCNN(nn.Module):
    def __init__(self):
        super(smallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)  # Output: 8 channels
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # Output: 16 channels
        self.pool = nn.MaxPool2d(2, 2)
        # Adjusted the input features of fc1 to 3456 (64x6x12)
        self.fc1 = nn.Linear(16*50*25, 512)
        self.fc2 = nn.Linear(512, 2) # For 2 classes
        self.dropout = nn.Dropout(0.2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output for the fully connected layer
        x = x.view(-1, 16*50*25)  # Adjusted to match the flattened feature maps
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.logsoftmax(x)
        return x