
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sleepiness.empty_seat.CNN.utils import transform

class FFNNEmptyfier(nn.Module):
    def __init__(self):
        super(FFNNEmptyfier, self).__init__()
        self.fc1 = nn.Linear(3 * 100 * 116, 512)  # Image size: 100x116
        self.fc2 = nn.Linear(512, 2)  # Output: 2 classes (empty, occupied)

    def forward(self, x):
        x = x.view(-1, 3 * 100 * 116)  # Image size: 100x116
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    train_dataset = torchvision.datasets.ImageFolder(root='pictures/empty_seat_dataset/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = torchvision.datasets.ImageFolder(root='pictures/empty_seat_dataset/test', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFNNEmptyfier().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    epochs = 5
    steps = 0
    running_loss = 0
    print_every = 500

    tr_loss = []
    val_loss = []
    print("Training started")
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            print(f"Step {steps}")
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    val_steps = 1
                    valbreak = len(val_loader)//30 # Only use 1/30 of the validation data for speed
                    for inputs, labels in val_loader:
                        if val_steps == valbreak:
                            break
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        batch_loss = criterion(outputs, labels)
                        
                        test_loss += batch_loss.item()
                        
                        _, predicted = outputs.max(dim=1)
                        equals = predicted == labels.view(*predicted.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        val_steps += 1
                        print(f"Validation step {val_steps}")
                
                tr_loss.append(running_loss/print_every)
                val_loss.append(test_loss/valbreak)
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f} | "
                    f"Test loss: {test_loss/valbreak:.3f} | "
                    f"Test accuracy: {accuracy/valbreak:.3f}")
                running_loss = 0
                model.train()
        # Make plots
        plt.plot(tr_loss, label='Training loss', color='#283618')
        plt.plot(val_loss, label='Validation loss', color='#bc6c25')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('empty_seat_loss.png', dpi=300)
        plt.close()
                
        # Save the model
        torch.save(model, f'sleepiness/empty_seat/FFNN/weights/empty_seat_epoch_{epoch+1}.pt')