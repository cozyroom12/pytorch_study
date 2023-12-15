#%% packages
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
# %% data import 
iris = load_iris()
X = iris.data 
y = iris.target

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#print(type(X_test)) # numpy array

# %% convert to float32 -> for... ?
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# %% dataset
class IrisData(Dataset):
    def __init__(self, X_train, y_train) :
        super().__init__()
        self.X = torch.from_numpy(X_train)
        self.y = torch.from_numpy(y_train)
        self.y = self.y.type(torch.LongTensor)  # what's LongTensor here?
        self.len = self.X.shape[0]

    def __getitem__(self, index):  # explain this function and how it works
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len

# %% dataloader
iris_data = IrisData(X_train, y_train)
train_loader = DataLoader(iris_data, batch_size=32)

# %% check dims
print(f"X shape: {iris_data.X.shape}, y shape: {iris_data.y.shape}")

# %% define class
class MultiClassNet(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
        super().__init__()
        self.lin1 = nn.Linear(NUM_FEATURES, HIDDEN_FEATURES)
        self.lin2 = nn.Linear(HIDDEN_FEATURES, NUM_CLASSES)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        x = self.log_softmax(x)
        return x

# %% hyper parameters
NUM_FEATURES = iris_data.X.shape[1]
HIDDEN = 6
NUM_CLASSES = len(iris_data.y.unique()) # tensor([0, 1, 2])

# %% create model instance 
model = MultiClassNet(NUM_FEATURES, NUM_CLASSES, HIDDEN)
# %% 아 얘네는 attribute 가 아닌건가 -> 맞는데 위에서 오타 내서 그럼
print(model.lin1)
# %% loss function
criterion = nn.CrossEntropyLoss()  # for classification

# %% optimizer
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# %% model.parameters() 뭐 나오더라
#print(model.parameters())
# %% training
NUM_EPOCHS = 100
losses = []
for epoch in range(NUM_EPOCHS):
    for x, y in train_loader: # x 는 features 고 y 는 라벨????인가
        # initialize gradients
        optimizer.zero_grad

        # forward pass
        y_hat_log = model(x)

        # calculate losses
        loss = criterion(y_hat_log, y)

        # calculate gradients
        loss.backward()

        # update parameters
        optimizer.step()
        
    losses.append(float(loss.data.detach().numpy()))
        
# %% show losses over each epoch
sns.lineplot(x=range(len(losses)), y=losses)
# %% test the model
X_test_torch = torch.from_numpy(X_test)
with torch.no_grad():
    y_test_hat_softmax = model(X_test_torch)
    #print(y_test_hat_softmax.data)
    y_test_hat = torch.max(y_test_hat_softmax.data, 1)  # .data attribute 는..
    #print(y_test_hat)

# %% Accuracy score from sklearn library
accuracy_score(y_test, y_test_hat.indices)

# %%
from collections import Counter
most_common_cnt = Counter(y_test).most_common()[0][1]
print(f"Naive Classifier: {most_common_cnt / len(y_test) * 100} %")
# %% save model state dict
