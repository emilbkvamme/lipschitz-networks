import torch
import torch.nn as nn
import monotonicnetworks as lmn

class UnconstrainedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth = 3):
        super(UnconstrainedModel, self).__init__()

        if depth < 2: # Check for valid depth
            raise ValueError("Depth must be at least 2 for the UnconstrainedModel.")

        layers = [ # Adding input layer
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        ]

        for _ in range(depth - 2): # Add hidden layers according to depth
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim)) # Output layer

        self.layers = nn.Sequential(*layers) # Create the sequential model with the defined layers
        
    def forward(self, x):
        return self.layers(x)


class LipschitzModel(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_groups=2,
                 depth = 3, 
                 lipschitz_bound=1.0):
        super(LipschitzModel, self).__init__()

        ######## CHECKING PARAMETERS #######
        if hidden_dim % n_groups != 0: # Check for divisibility for the GroupSort activation
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_groups ({n_groups})")
        
        if depth < 2: # Check for valid depth
            raise ValueError("Depth must be at least 2 for the LipschitzModel.")

        ####### ADDING LAYERS #######
        layers = [ # Adding input layer with direct normalization
            lmn.direct_norm(torch.nn.Linear(input_dim, hidden_dim), kind="one-inf"),
            lmn.GroupSort(n_groups)
        ]

        for _ in range(depth - 2): # Add hidden layers according to depth
            layers.append(lmn.direct_norm(nn.Linear(hidden_dim, hidden_dim), kind="inf"))
            layers.append(lmn.GroupSort(n_groups))

        layers.append(lmn.direct_norm(nn.Linear(hidden_dim, output_dim), kind="inf")) # Output layer

        self.layers = nn.Sequential(*layers) # Create the sequential model with the defined layers
        self.lipschitz_bound = lipschitz_bound  # Store the Lipschitz bound for later use
        
    def forward(self, x):
        return self.layers(x)*self.lipschitz_bound  