from torch import nn

# Set up custom CNN Network
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(16*7*15, 32)
        self.fc3 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    # Different impl with batch norm
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16*7*15)
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc3(x)
        x = self.logsoftmax(x)
        return x