
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Create a toy dataset: y = 2x + 1 with some noise
torch.manual_seed(42)
data_size = 1000
X = torch.linspace(-10, 10, data_size).reshape(-1, 1)
Y = 2 * X + 1 + torch.randn(X.size()) * 2

# Splitting data into training and validation
train_size = int(0.8 * data_size)
val_size = data_size - train_size
train_data, val_data = random_split(list(zip(X, Y)), [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=val_size)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.fc(x)

# Initialize model, criterion, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def evaluate_model(loader):
    """Evaluate the model's performance on a given dataloader."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

# Implementing the Intelligent Epoch
prev_val_loss = float('inf')
max_epochs = 100
tolerance = 5e-3  # Minimum improvement in validation loss to continue training
for epoch in range(max_epochs):
    
    # Self-evaluation: Check performance on validation set
    val_loss = evaluate_model(val_loader)
    performance_improvement = prev_val_loss - val_loss
    
    # Self-termination: If improvement is below a threshold, stop training
    if performance_improvement < tolerance:
        break
    
    # Strategic planning: If performance drops or improvement is marginal, reduce learning rate
    if performance_improvement < 2 * tolerance:
        for g in optimizer.param_groups:
            g['lr'] *= 0.5  # Halve the learning rate
    
    # Adaptive learning: For simplicity, we'll keep the data sampling uniform in this example
    
    # Training loop
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    prev_val_loss = val_loss

print(f"Trained for {epoch} epochs. Final model parameters: Weight = {model.fc.weight.item()}, Bias = {model.fc.bias.item()}")
