import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

## heart attack dataset from kaggle
df = pd.read_csv('../dataset/heart.csv')
#print(df.head())

# separate independent / dependent features
X = np.array(df.loc[:, df.columns != 'output'])
y = np.array(df['output'])
print(f"X: {X.shape}, y:{y.shape}")  # X: (303, 14), y: (303,)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Neural Network Class
class NeuralNetworkFromScratch:
    def __init__(self, LR, X_train, y_train, X_test, y_test):
        self.w = np.random.randn(X_test_scaled.shape[1])
        self.b = np.random.randn() # just one value for bias 
        self.LR = LR
        self.X_train = X_train 
        self.y_train = y_train
        self.X_test = X_test 
        self.y_test = y_test
        self.L_train = []  # train loss
        self.L_test = []  # test loss 

    # helper functions
    def activation(self, x):
        # sigmoid 
        return 1 / (1 + np.exp(-x))
    
    # derivative of sigmoid -> i forgot why this was needed??
    def dactivation(self, x): 
        return self.activation(x) * (1 - self.activation(x))
    
    # forward pass
    def forward(self, X):
        hidden_1 = np.dot(X, self.w) + self.b 
        activate_1 = self.activation(hidden_1)
        return activate_1
    
    # backward pass
    def backward(self, X, y_true):
        # calculate gradients
        hidden_1 = np.dot(X, self.w) + self.b
        y_pred = self.forward(X)
        dL_dpred = 2 * (y_pred - y_true)
        dpred_dhidden1 = self.dactivation(hidden_1)
        dhidden1_db = 1
        dhidden_dw = X

        dL_db = dL_dpred * dpred_dhidden1 * dhidden1_db
        dL_dw = dL_dpred * dpred_dhidden1 * dhidden_dw
        return dL_db, dL_dw 
    
    # optimizer 
    def optimizer(self, dL_db, dL_dw):
        # update weights
        self.b = self.b - dL_db * self.LR 
        self.w = self.w - dL_dw * self.LR 

    # training
    def train(self, ITER):
        for i in range(ITER):
            # random position setting
            random_pos = np.random.randint(len(self.X_train))

            # forward pass
            y_train_true = self.y_train[random_pos]
            y_train_pred = self.forward(y_train[random_pos])

            # calculate training losses
            L = np.sum(np.square(y_train_pred - y_train_true))
            self.L_train.append(L)

            # calculate gradients
            dL_db, dL_dw = self.backward(self.X_train[random_pos], y_train[random_pos])

            # update weights
            self.optimizer(dL_db, dL_dw)

            # caculate error for test data
            L_sum = 0
            for j in range(len(self.X_test)):
                y_true = self.y_test[j]
                y_pred = self.forward(self.X_test[j])
                L_sum += np.square(y_pred - y_true)
            self.L_test.append(L_sum)
        return "training successful"
    
# hyperparameters
LR = 0.1
ITER = 1000 

nn = NeuralNetworkFromScratch(LR, X_train_scaled, y_train, X_test_scaled, y_test)
nn.train(ITER)

# ----- #
# check losses using seaborn
sns.lineplot(x = list(range(len(nn.L_test))), y = nn.L_test)

