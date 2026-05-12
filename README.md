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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SGDClassifier(max_iter=1000, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

```
## Output:
<img width="630" height="288" alt="image" src="https://github.com/user-attachments/assets/0e65a267-2bb8-46cc-b3d4-e204f7e9499b" />

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
