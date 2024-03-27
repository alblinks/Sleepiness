import numpy as np
import torch
import torchvision
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sleepiness.face.smallCNN.transforms import train_transform as transform

from sleepiness.face.smallCNN.weights import __path__ as smallcnn_WeightPath
from sleepiness.face.smallCNN.model import smallCNN


# Data loading
train_dataset = torchvision.datasets.ImageFolder(root="pictures/notcorrect_faces/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,num_workers=2)

val_dataset = torchvision.datasets.ImageFolder(root="pictures/notcorrect_faces/test", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True,num_workers=2)

model = smallCNN()
# Set the device
print("Setting the device")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Train the model
epochs = 500
steps = 0
running_loss = 0
print_every = 100

tr_loss = [1]
val_loss = [1]
accuracies = [0]
val_rounds = 0
print("Training started")

for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1

        #if steps % 50 == 0:
        #    print(f"Step {steps}")

        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            val_rounds += 1
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
                    #print(f"Validation step {val_steps}")

            tr_loss.append(running_loss/print_every)
            val_loss.append(test_loss/valbreak)
            accuracies.append(accuracy/valbreak)

            # Make plots
            step_count = np.arange(0, (val_rounds+1)*print_every, print_every)
            plt.plot(step_count,tr_loss, label='Training loss', color='#283618')
            plt.plot(step_count,val_loss, label='Validation loss', color='#bc6c25')
            plt.plot(step_count,accuracies, label='Validation accuracy', color='#f0a500')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('small_face_cnn_loss.png', dpi=300)
            plt.close()
     
            print(f"Epoch {epoch+1}/{epochs}.. ",
                  f"Train loss: {running_loss/print_every:.3f} | ",
                  f"Test loss: {test_loss/valbreak:.3f} | ",
                  f"Test accuracy: {accuracy/valbreak:.3f}")
            running_loss = 0
            model.train()
            
    # Save the model
    torch.save(model, f'{smallcnn_WeightPath[0]}/small_face_epoch_{epoch+1}.pt')
