"""
Train Heart Disease Prediction Models
This script trains 5 ML algorithms: Logistic Regression, SVM, Decision Tree, Random Forest, KNN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import pickle

print("="*60)
print("Starting Heart Disease Prediction Model Training")
print("="*60)

# Read Dataset
print("\n1. Loading dataset...")
heartData = pd.read_csv("./dataset/heart_data_set.csv")
print(f"   Dataset loaded: {heartData.shape[0]} rows, {heartData.shape[1]} columns")

# Check for missing data
missing_data = heartData.isnull().sum()
total_percentage = (missing_data.sum()/heartData.shape[0]) * 100
print(f"   Missing data: {round(total_percentage, 2)}%")

# Prepare features and target
print("\n2. Preparing features and target...")
x_data = heartData.drop(['target'], axis=1)
y_data = heartData.target.values
print(f"   Features shape: {x_data.shape}")
print(f"   Target shape: {y_data.shape}")

# Split data
print("\n3. Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
print(f"   Train set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Feature Scaling
print("\n4. Applying feature scaling...")
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("   Scaling completed")

# Train Models
print("\n5. Training Machine Learning Models...")
print("="*60)

# 1. K-Nearest Neighbors
print("\n   [1/5] Training K-Nearest Neighbors (KNN)...")
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
knn_prediction = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_prediction)
print(f"         KNN Accuracy: {knn_accuracy:.4f} ({knn_accuracy*100:.2f}%)")

# 2. Decision Tree
print("\n   [2/5] Training Decision Tree Classifier...")
dt_classifier = DecisionTreeClassifier(random_state=0)
dt_classifier.fit(X_train, y_train)
dt_prediction = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_prediction)
print(f"         Decision Tree Accuracy: {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")

# 3. Random Forest
print("\n   [3/5] Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=10, random_state=0)
rf_model.fit(X_train, y_train)
rf_prediction = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_prediction)
print(f"         Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")

# 4. Support Vector Machine
print("\n   [4/5] Training Support Vector Machine (SVM)...")
svc_classifier = SVC(random_state=0)
svc_classifier.fit(X_train, y_train)
svc_prediction = svc_classifier.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_prediction)
print(f"         SVM Accuracy: {svc_accuracy:.4f} ({svc_accuracy*100:.2f}%)")

# 5. Logistic Regression
print("\n   [5/5] Training Logistic Regression...")
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
logistic_model_prediction = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_model_prediction)
print(f"         Logistic Regression Accuracy: {logistic_accuracy:.4f} ({logistic_accuracy*100:.2f}%)")

# Save all models
print("\n" + "="*60)
print("6. Saving trained models to models.pkl...")
all_models = [rf_model, logistic_model, dt_classifier, svc_classifier, knn_classifier]
with open("models.pkl", 'wb') as files:
    pickle.dump(all_models, files)
print("   Models saved successfully!")

# Verify saved models
print("\n7. Verifying saved models...")
with open("models.pkl", "rb") as open_file:
    loaded_list = pickle.load(open_file)
print(f"   Loaded {len(loaded_list)} models from models.pkl")
print("   Models: [Random Forest, Logistic Regression, Decision Tree, SVM, KNN]")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)
print("\nModel Accuracies Summary:")
print(f"  1. Random Forest:        {rf_accuracy*100:.2f}%")
print(f"  2. Logistic Regression:  {logistic_accuracy*100:.2f}%")
print(f"  3. Decision Tree:        {dt_accuracy*100:.2f}%")
print(f"  4. SVM:                  {svc_accuracy*100:.2f}%")
print(f"  5. KNN:                  {knn_accuracy*100:.2f}%")
print("\nYou can now run the Flask app using: python app.py")
print("="*60)
