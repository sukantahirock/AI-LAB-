import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
data_set = pd.read_csv('dataset.csv')

# Debug: print the shape and columns of the DataFrame
print("Shape of dataset:", data_set.shape)
print("Columns in dataset:", data_set.columns)

# Data preprocessing
# Check for missing values
print("Missing values in each column:\n", data_set.isnull().sum())

# Assuming no missing values for now
# Extracting Independent and dependent Variable
try:
    X = data_set.iloc[:, [1,2]].values
    y = data_set.iloc[:, 3].values
except IndexError as e:
    print("IndexError:", e)
    print("Please check the column indices and ensure they are within the bounds of the DataFrame.")
    raise

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Define models with pipelines for scaling (if needed)
models = [
    ('Logistic Regression', Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Support Vector Machine', Pipeline([('scaler', StandardScaler()), ('clf', SVC())]))
]

# Initialize list to store test accuracies
test_accuracies = []

# Evaluate models
for name, model in models:
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Model: {name}")
    print("Cross-validation scores:", cv_scores)
    print("Mean accuracy:", np.mean(cv_scores))
    
    # Fit the model on the entire training set
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate accuracy on the test set
    test_accuracy = accuracy_score(y_test, y_pred)
    test_accuracies.append((name, test_accuracy))
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate classification report
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix for {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # Plot predicted vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title(f"Predicted vs Actual for {name}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()
    
    print("\n")

# Plot test accuracies
model_names, accuracies = zip(*test_accuracies)
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.xlabel('Model')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy of Different Models')
plt.ylim(0, 1)
plt.show()

# Summary table of model performances
summary_df = pd.DataFrame(test_accuracies, columns=['Model', 'Test Accuracy'])
print(summary_df)

