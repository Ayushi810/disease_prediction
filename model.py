import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")

# Split data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier()

# Train
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)

# Predictions
lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)

# Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

# Confusion Matrix
print("LR Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print("DT Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))

# Save best model (example: logistic regression)
pickle.dump(lr, open("model.pkl", "wb"))