# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1. Import Necessary Libraries and Load Data.

 2. Split Dataset into Training and Testing Sets.

 3. Train the Model Using Stochastic Gradient Descent (SGD).

 4. Make Predictions and Evaluate Accuracy.

 5. Generate Confusion Matrix
    
## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: VAISHNAVI T
RegisterNumber:  212225040479
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
data = pd.read_csv("Placement_Data.csv")
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})
X = data[['ssc_p', 'mba_p']].values
y = data['status'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
m = len(y)
X = np.c_[np.ones(m), X]
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    h = np.clip(h, 1e-10, 1 - 1e-10)   # prevents log(0)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))
theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []

for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    
    cost = cost_function(X, y, theta)
    cost_history.append(cost)
y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)
accuracy = np.mean(y_pred == y) * 100
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y, y_pred))
```
## Output:
<img width="580" height="290" alt="Screenshot 2026-05-11 115625" src="https://github.com/user-attachments/assets/e4b98bac-e4c9-4642-a824-481f6b9e5577" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
