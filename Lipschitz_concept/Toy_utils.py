# Utility functions for the toy example notebook
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


##### DATASET GENERATION #####
def sine_function_train_data(N, frequency=1.0, seed=6):
    """Generating the sine dataset with noise. Based on the LHCb paper."""
    torch.manual_seed(seed) # Setting seed for reproducibility

    x_train = torch.linspace(-2*np.pi, 2*np.pi, N).unsqueeze(-1)
    
    y_train_error = torch.ones(N).unsqueeze(-1) * 0.01 
    y_train_error[N // 2] = 1
    y_train = torch.sin(frequency * x_train) + torch.randn(N).view(-1, 1) * y_train_error # Adding noise to increase learning

    # Plot data
    x_linspace = torch.linspace(-2*np.pi, 2*np.pi, 2*N).unsqueeze(-1)
    y_linspace = torch.sin(frequency * x_linspace)

    return x_train, y_train, x_linspace, y_linspace

def gaussian_function_data(N, mu, sigma, seed=6, noise_magnitude=0.01):
    """
    Generate data from a Gaussian function with added noise.
    """
    torch.manual_seed(seed)  # Setting seed for reproducibility

    x_train = torch.linspace(-3*sigma, 3*sigma, N).unsqueeze(-1)

    # Adding some noise to the Gaussian function to make it more challenging for the model to learn
    train_error = torch.randn(N).view(-1, 1) * noise_magnitude # Sampling N random numbers from a normal distribution
    gaussian = (1/(sigma * torch.sqrt(torch.tensor(2 * np.pi)))) * torch.exp(-0.5 * ((x_train - mu) / sigma) ** 2)
    y_train = gaussian +  train_error  # Adding noise to increase learning

    # Plotting data
    x_linspace = torch.linspace(-3*sigma, 3*sigma, 2*N).unsqueeze(-1)
    y_linspace = (1/(sigma * torch.sqrt(torch.tensor(2 * np.pi)))) * torch.exp(-0.5 * ((x_linspace - mu) / sigma) ** 2)
    return x_train, y_train, x_linspace, y_linspace


class BasicDataset(Dataset):
    """A simple dataset class for handling input and target data."""
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

###### LIPSHITZ CALCULATIONS ######
# Lipschitz constant of the gaussian
def gaussian_lipschitz_constant(sigma):
    """
    Calculating the Lipschitz constant of a Gaussian function.
    Do not need to care about the mu as translations do no change the gradients of a function
    """
    sqrt_2_pi_e = np.sqrt(2 * np.pi*np.e)
    sigma_squared = sigma ** 2
    return np.abs(-1/(sigma_squared * sqrt_2_pi_e))


def calculate_lipschitz_constant(model, x_data):
    """Calculates the Lipschitz constants from the prediction of a model. This is done by computing the Jacobian of the models output wrt its input, and calculating its maximum singular value which correspondst to the Lipschitz constant.
    
    NOTE: works for small datasets. Larger datasets should be subsampled to avoid memory issues. Potential local estimates can be done as well.
    """
    model.eval()
    with torch.no_grad():
        # Compute the Jacobian of the model's output with respect to its input
        x_data = x_data.clone().detach().requires_grad_(True)  # Ensure x_data requires gradients
        y_pred = model(x_data)
        jacobian = torch.autograd.functional.jacobian(model, x_data)

        # Calculate the Lipschitz constant as the maximum singular value of the Jacobian
        lipschitz_constant = torch.linalg.norm(jacobian, ord=2, dim=(1, 2)).max().item()

    return lipschitz_constant

def lipschitz_floor_1D(x, y, L, max_points = 5000):
    """ Best fit possible for an L-Lipschitz function to the points (x, y) in 1D. Done by solving a quadratic programming problem.
    
    Returns
        g : best L-Lipschitz values at the input points
        mse : the floor, mean((g - y)^2)
    
    """

#### OPTIMIZING THE FLOOR ####


