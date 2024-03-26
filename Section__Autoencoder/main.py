#%%
from typing import OrderedDict
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt 
import torchvision.utils 
# %%
path_images = 'data/train'

transform = transforms.Compose(
    [transforms.Resize((64,64)),
     transforms.Grayscale(num_output_channels=1),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]
)

dataset = ImageFolder(root=path_images, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
# %% model class
LATENT_DIMS = 128

# Implement Encoder class

# Implement Decoder class

# Implement Autoencoder class

# Test
# input = torch.rand((1, 1, 64, 64))