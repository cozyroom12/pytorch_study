#%% packages
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# %% Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
# %%
print(len(y))
# %% Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make prediction on the test set 
predictions = model.predict(X_test)

# Evaluate accuracy 
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# %% test
iris = load_iris()
data = iris.data
target = iris.target 
target_names = iris.target_names
feature_names = iris.feature_names
description = iris.DESCR 
print("Feature Matrix (data):\n", data[:10])
print("Target Variable (target):\n", target[:10])
print("Target Names (target_names):\n", target_names)
print("Feature Names (feature_names):\n", feature_names)
print("Description (DEScR):\n", description)
# %%
