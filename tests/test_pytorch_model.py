import unittest
import torch
import torch.nn as nn

# Define the SimpleNN class
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, X):
        return self.linear(X)

# Define the unit test class
class TestSimpleNN(unittest.TestCase):
    def test_pytorch_model(self):
        X = torch.tensor([[1, 1], [2, 2], [3, 3]], dtype=torch.float32)
        y = torch.tensor([2, 4, 6], dtype=torch.float32).view(-1, 1)

        model = SimpleNN(input_size=2, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Training loop
        for epoch in range(1000):
            predictions = model(X)
            loss = criterion(predictions, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Predictions after training
        with torch.no_grad():
            predicted = model(X)
        
        # Assert that predictions are close to actual values
        self.assertTrue(torch.allclose(predicted, y, rtol=0.1))

if __name__ == "__main__":
    unittest.main()

