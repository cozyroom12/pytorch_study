#%% packages
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns 
import numpy as np
from collections import Counter 

# %% data prep 
X, y = make_multilabel_classification(n_samples=10000, )
