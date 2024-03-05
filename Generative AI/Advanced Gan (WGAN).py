#%% 
import torch, torchvision, os, PIL, pdb 
from torch import nn 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from torchvision import transforms 
from torchvision.utils import make_grid 
from tqdm.auto import tqdm 
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 

def show(tensor, num=25, name=''):
    data = tensor.detach().cpu()
    grid = make_grid(data[:num], nrow=5).permute(1,2,0) # plt

    plt.imshow(grid.clip(0,1))
    plt.show()

### hyperparameters and general parameters
n_epochs = 10000
batch_size = 128
lr = 1e-4
z_dim = 200
device = 'cuda'

cur_step = 0
crit_cycles = 5
gen_losses = []
crit_losses = []
show_step = 35 
save_step = 35  # checkpoint save
# %% generator model

class Generator(nn.Module):
    def __init__(self, z_dim=64, d_dim)