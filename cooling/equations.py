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

def cooling_equation(time, Tenv, T0, R):
    """
    Compute the temperature at a given time based on Newton's Law of Cooling.

    Args:
        time: Time at which to compute the temperature (scalar or array-like).
        Tenv: Ambient temperature (scalar).
        T0: Initial temperature of the object (scalar).
        R: Cooling rate constant (scalar).
    
    Returns:
        Temperature at the given time.
    """
    return Tenv + (T0 - Tenv) * np.exp(-R * time)