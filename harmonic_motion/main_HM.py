"""
@author:; Michele Carletti
@description: Neural network for modeling damped harmonic motion.
"""

import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from networks import PINN
from equations import grad, hm_equation

sns.set_theme()
torch.manual_seed(42)
np.random.seed(10)

# Physical parameters for harmonic motion
m = 1.0  # mass (kg)
k = 1.0  # spring constant (N/m)
mu = 0.3  # damping coefficient (kg/s)
eq = partial(hm_equation, m=m, mu=mu, k=k)  # damped HM
times = np.linspace(0, 15, 1000)
pos = eq(times)  # Position function

# Generate training data
t_data = np.linspace(0, 10, 20)  # time from 0 to 10 seconds

# Generate noisy observations of position
x_data = eq(t_data) + 0.1 * np.random.randn(len(t_data))

def physics_loss_shm(model: torch.nn.Module):
    """
    Compute the physics-based loss for harmonic motion.
    """
    ts = torch.linspace(0, 15, 1000).view(-1, 1).requires_grad_(True)
    positions = model(ts)   # Predict positions
    velocities = grad(positions, ts)[0] # Compute velocities
    accelerations = grad(velocities, ts)[0] # Compute accelerations
    # Harmonic Motion equation: m * d²x/dt² + mu * dx/dt + k * x = 0
    pde = m * accelerations + mu * velocities + k * positions

    # Enforce initial condition to avoid trivial solution x(t)=0
    t0 = torch.tensor([[0.0]], requires_grad=True)
    x0 = model(t0)
    v0 = grad(x0, t0)[0]
    bc_loss = (x0 - 1.0)**2 + v0**2  # x(0)=1, v(0)=0

    return torch.mean(pde**2) + torch.mean(bc_loss)

if __name__ == "__main__":
    # Convert training data to tensors
    t_tensor = torch.tensor(t_data, dtype=torch.float32).view(-1, 1)
    x_tensor = torch.tensor(x_data, dtype=torch.float32).view(-1, 1)

    # Initialize and train the PINN model
    model = PINN(
        input_dim=1,
        out_dim=1,
        n_units=150,
        epochs=16000,
        lr=1e-3,
        py_loss=physics_loss_shm,
        py_weight=10.0
    )

    losses = model.fit(t_tensor, x_tensor)

    # Predict positions over a fine time grid
    t_fine = np.linspace(0, 15, 1000)
    preds = model.predict(torch.tensor(t_fine, dtype=torch.float32).view(-1, 1)).detach().numpy()

    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    # Plot predictions vs true solution
    plt.figure(figsize=(10, 4))
    plt.plot(times, pos, label="True Solution", color="green")
    plt.scatter(t_data, x_data, label="Noisy Observations", color="red")
    plt.plot(t_fine, preds, label="PINN Prediction", color="blue", linestyle="--")
    plt.title("Simple Harmonic Motion: PINN Prediction vs True Solution")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.show()