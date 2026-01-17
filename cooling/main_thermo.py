"""
@author: Michele Carletti
@description: Using PINN to model thermodynamic cooling process.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

from equations import cooling_equation, grad
from networks import PINN

sns.set_theme()
torch.manual_seed(42)
np.random.seed(10)

Tenv = 25.0  # Ambient temperature in °C
T0 = 100.0  # Initial temperature in °C
R = 0.005   # Cooling rate constant
times = np.linspace(0, 1000, 1000)  # Time from 0 to 1000 seconds
eq = partial(cooling_equation, Tenv=Tenv, T0=T0, R=R)   # Define the equation with fixed parameters
temps = eq(times) # Compute temperatures

# Prepare training data
t = np.linspace(0, 300, 15) # Training times from 0 to 300 seconds
T = eq(t) + 2 * np.random.randn(15)  # Corresponding temperatures with noise

def physics_loss(model: torch.nn.Module):
    """
    Compute the physics-based loss for the cooling equation.
    """
    ts = torch.linspace(0, 1000, 1000).view(-1, 1).requires_grad_(True)
    temps = model(ts)
    dT = grad(temps, ts)[0]
    pde = R * (Tenv - temps) - dT
    return torch.mean(pde**2)

if __name__ == "__main__":
    # Convert training data to tensors
    t_tensor = torch.tensor(t, dtype=torch.float32).view(-1, 1)
    T_tensor = torch.tensor(T, dtype=torch.float32).view(-1, 1)

    # Initialize and train the PINN model
    model = PINN(
        input_dim=1,
        out_dim=1,
        n_units=50,
        epochs=5000,
        lr=1e-3,
        py_loss=physics_loss,
        py_weight=1.0
    )

    losses = model.fit(t_tensor, T_tensor)

    preds = model.predict(torch.tensor(times, dtype=torch.float32).view(-1, 1)).detach().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.yscale("log")
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(times, temps, alpha=0.8)
    plt.plot(t, T, 'o')
    plt.plot(times, preds, alpha=0.8)
    plt.legend(["Equation", "Training Data", "PINN Prediction"])
    plt.title("Thermodynamic Cooling Process: PINN vs Equation")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.show()