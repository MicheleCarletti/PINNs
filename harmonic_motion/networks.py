"""
@author: Michele Carletti
@description: Physics-Informed Neural Network (PINN) implementation using PyTorch.
"""
import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
    def __init__(
            self,
            input_dim,
            out_dim,
            n_units=100,
            epochs=1000,
            loss=nn.MSELoss(),
            lr=1e-3,
            py_loss=None,
            py_weight=0.1
    ) ->None:
        
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.lr = lr
        self.py_loss = py_loss
        self.py_weight = py_weight
        self.n_units = n_units

        self.model = nn.Sequential(
            nn.Linear(input_dim, n_units),
            nn.Tanh(),  # Here we use Tanh activation for better performance in SHM
            nn.Linear(n_units, n_units),
            nn.Tanh(),
            nn.Linear(n_units, out_dim)
        )
    
    def forward(self, x):
        return self.model(x)
    
    def fit(self, x, y):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.forward(x)
            loss = self.loss(outputs, y)

            if self.py_loss:
                loss += self.py_weight * self.py_loss(self)
            
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')

        return losses
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)
