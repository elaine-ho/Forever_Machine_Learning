import torch

def print_model(model, x):
    for layer in model:
        x = layer(x)
        print(layer.__class__.__name__,'output shape: \t',x.shape)
