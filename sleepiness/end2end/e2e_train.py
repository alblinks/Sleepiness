import torch
import torchvision
from torchvision import models
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet152_Weights
from sleepiness.end2end.utils import transform


train_dataset = torchvision.datasets.ImageFolder(root='path/to/train_folder', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

val_dataset = torchvision.datasets.ImageFolder(root='path/to/test_folder', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Load the pre-trained model
print("Loading pre-trained model")
model = models.resnet152(weights = ResNet152_Weights.DEFAULT)

# Freeze the pre-trained layers
for param in model.parameters():
    param.requires_grad = False
    
# Replace the last fully connected layer with a new one
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 2),
    nn.LogSoftmax(dim=1)
)

# Set the device
print("Setting the device")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

# Train the model
epochs = 30
steps = 0
running_loss = 0
print_every = 100

tr_loss = []
val_loss = []
print("Training started")
for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1
        print(f"Step {steps}")
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                val_steps = 1
                valbreak = len(val_loader)//3 # Only use 1/3 of the validation data for speed
                for inputs, labels in val_loader:
                    if val_steps == valbreak:
                        break
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    val_steps += 1
                    print(f"Validation step {val_steps}")
               
            tr_loss.append(running_loss/print_every)
            val_loss.append(test_loss/valbreak)
            
            # Make plots
            plt.plot(tr_loss, label='Training loss', color='#283618')
            plt.plot(val_loss, label='Validation loss', color='#bc6c25')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('e2e_loss.png', dpi=300)
                 
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f} | "
                  f"Test loss: {test_loss/valbreak:.3f} | "
                  f"Test accuracy: {accuracy/valbreak:.3f}")
            running_loss = 0
            model.train()
            
    # Save the model
    torch.save(model, f'e2e_epoch_{epoch+1}.pt')
