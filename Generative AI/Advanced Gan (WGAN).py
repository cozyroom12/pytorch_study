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

def show(tensor, num=25, wamb=0, name):