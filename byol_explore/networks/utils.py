import torch.nn as nn

def create_mlp(inp_size, num_hidden, num_units, out_size):
    layers = [nn.Linear(inp_size, num_units), nn.GELU()]
    for _ in range(num_hidden - 1):
        layers.append(nn.Linear(num_units, num_units))
        layers.append(nn.GELU())
    
    layers.append(nn.Linear(num_units, out_size))
    
    return nn.Sequential(*layers) 

