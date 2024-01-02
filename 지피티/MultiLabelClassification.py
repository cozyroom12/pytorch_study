#%% packages 
from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split 
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report 
# %% Load Reuters dataset 
reuters = fetch_openml(name='reuters-21578', version=1)
X, y = reuters.data, reuters.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a multi-label logistic regression model
model = OneVsRestClassifier(LogisticRegression())
model.fit(X_train, y_train)

# Make predictions on the test set 
predictions = model.predict(X_test)

# Evaluate accuracy and other metrics 
acc = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
print(f"Acc: {acc}")
print("Classification Report:\n", report)
# %% OpenMLError: Dataset reuters-21578 with version 1 not found. 에러
import nltk 
from nltk.corpus import reuters 
