import torch.nn as nn

def create_mlp(inp_size, num_hidden, num_units, out_size, dropout_prob=0.0):
    layers = [nn.Linear(inp_size, num_units), nn.GELU()]
    if dropout_prob > 0:
        layers.append(nn.Dropout(dropout_prob))
    for _ in range(num_hidden - 1):
        layers.append(nn.Linear(num_units, num_units))
        layers.append(nn.GELU())
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))

    layers.append(nn.Linear(num_units, out_size))
    
    return nn.Sequential(*layers) 

