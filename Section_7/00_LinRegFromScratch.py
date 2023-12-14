#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns

# %% data import
cars_file = "https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv"
cars = pd.read_csv(cars_file)
cars.head()

# %% visualize
sns.scatterplot(x='wt',y='mpg',data=cars)
sns.regplot(data=cars, x='wt', y='mpg') # regression line

# %% convert data to tensor 
X_list = cars.wt.values 


X_np = np.array(X_list, dtype=np.float32).reshape(-1, 1)
# 여기는 .tolist() 를 쓴거지 왜
y_list = cars.mpg.values.tolist()

# two ways of tensorifying lol
X = torch.from_numpy(X_np)
y = torch.tensor(y_list)
print(f"X shape: {X.shape} and Y shape: {y.shape}")

# %% training 
w = torch.rand(1, requires_grad=True, dtype=torch.float32) # just one random value
b = torch.rand(1, requires_grad=True, dtype=torch.float32)

num_epochs = 1000
learning_rate = 0.001

for epoch in range(num_epochs):
    for i in range(len(X)): # batch size of one?
        # forward pass
        y_pred = X[i] * w + b 
        # calculate loss
        loss_tensor = torch.pow(y_pred - y[i], 2)

        # backward pass
        loss_tensor.backward()

        # extract losses
        loss_value = loss_tensor.data[0]

        # update weights and biases
        with torch.no_grad():
            w -= w.grad * learning_rate
            b -= b.grad * learning_rate
            w.grad.zero_() # inplace operation (_) ? whaaat
            b.grad.zero_()
    print(loss_value)

# %% check results 
print(f"Weight: {w.item()}, Bias: {b.item()}")

# %%
y_pred = ((X * w) + b).detach().numpy()
y_pred
# %% visualize
sns.scatterplot(x=X_list, y=y_list)
sns.lineplot(x=X_list, y=y_list)
sns.lineplot(x=X_list, y=y_pred.reshape(-1))

# %% (Statistical) Linear Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_np, y_list)
print(f"Slope: {reg.coef_}, Intercept: {reg.intercept_}")

# %% bonus !! graph visualization
# download graphviz
# i will skip this part

# %%
