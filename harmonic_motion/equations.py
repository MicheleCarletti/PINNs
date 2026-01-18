"""
@author: Michele Carletti
@description: Cooling equations and related physical constants.
"""
import numpy as np
import torch

def grad(outputs, inputs):
    """
    Compute the partial derivative of 'outputs' with respect to 'inputs'.

    Args:
        outputs: Tensor (N, 1)
        inputs: Tensor (N, D)
    
    """
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
    )


def hm_equation(time, m, mu, k):
    """
    Compute the position of a damped harmonic oscillator at a given time.

    Args:
        time: Time at which to compute the position (scalar or array-like).
        m: Mass of the object (scalar).
        mu: Damping coefficient (scalar).
        k: Spring constant (scalar).
    
    Returns:
        Position at the given time.
    """
    omega_0 = np.sqrt(k / m)
    gamma = mu / (2 * m)
    omega_d = np.sqrt(omega_0**2 - gamma**2)

    return np.exp(-gamma * time) * np.cos(omega_d * time)

        