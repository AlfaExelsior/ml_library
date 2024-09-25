import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, X):
        return self.linear(X)

# Usage
X = torch.tensor([[1, 1], [2, 2], [3, 3]], dtype=torch.float32)
y = torch.tensor([2, 4, 6], dtype=torch.float32).view(-1, 1)

model = SimpleNN(input_size=2, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predictions
with torch.no_grad():
    print(model(X))
