# elevatedhealth1.io
import pandas as pd

from sklearn.model_selection import train_test_split from sklearn.naive_bayes import GaussianNB from sklearn.metrics import accuracy_score, classification_report, confusion_matrix from sklearn.datasets import load_iris

# Load the Iris dataset

iris load_iris()

X = iris.data # Features

yiris.target # Target classes

# Split the dataset into training and test sets (60% training, 40% testing) X_train, X_test, y_train, y_test train_test_split(X, y, test_size=0.4, random_state=42)

# Initialize the Naive Bayes classifier (GaussianNB for continuous features)

model GaussianNB()

# Train the classifier model.fit(X_train, y_train)

# Predict on test data y_pred model.predict(X_test)

# Calculate accuracy

accuracy accuracy_score (y_test, y_pred)

# Display results

print("Predicted Labels:", y_pred) print("Actual Labels v test)
Display results

print("Predicted Labels:", y_pred)

print("Actual Labels :", y_test)

print("\nAccuracy of Naive Bayes classifier: {:.2f}%".format(accuracy 100))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

Predicted Labels: [102110121120000 2 2 1 1 2 0 2 0 2