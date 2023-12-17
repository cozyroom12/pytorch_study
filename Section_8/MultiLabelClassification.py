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
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# %% 
test = X_torch.shape
print(test)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size=0.2)

# %% dataset and dataloader
class MultilabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
multilabel_data = MultilabelDataset(X_train, y_train)
train_loader = DataLoader(dataset=multilabel_data, batch_size=10)
# %% model
# topology: fc1, relu, fc2
# final activation function??
class MultilabelNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultilabelNetwork, self).__init__()
        self.fc1 =nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x 
    
input_dim = X_torch.shape[1]
output_dim = y_torch.shape[0]
model = MultilabelNetwork()
# %%
