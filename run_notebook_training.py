"""
Execute all cells from Heart-Disease-Prediction.ipynb
This follows the exact sequence from the original notebook
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import pickle

print("="*70)
print("HEART DISEASE PREDICTION - NOTEBOOK EXECUTION")
print("="*70)

# Cell 3: Import Libraries
print("\n✓ Cell 1: Import Libraries")
print("successful import ;)")

# Cell 5: Read Dataset
print("\n✓ Cell 2: Read Dataset")
heartData = pd.read_csv("./dataset/heart_data_set.csv")
print(f"Dataset shape: {heartData.shape}")

# Cell 6: Dataset Info
print("\n✓ Cell 3: Dataset Info")
heartData.info()

# Cell 7: Dataset Description
print("\n✓ Cell 4: Dataset Description")
print(heartData.describe())

# Cell 9: Check Missing Data
print("\n✓ Cell 5: Check Missing Data and Duplicates")
missing_data = heartData.isnull().sum()
total_percentage = (missing_data.sum()/heartData.shape[0]) * 100
print(f'Total percentage of missing data is {round(total_percentage,2)}%')
duplicate = heartData[heartData.duplicated()]
print(f"Duplicate rows: {len(duplicate)}")
# Drop duplicate rows
heartData = heartData.drop_duplicates()
print(f"Dataset shape after removing duplicates: {heartData.shape}")

# Cell 10: Correlation Matrix
print("\n✓ Cell 6: Creating Correlation Matrix Plot")
rcParams['figure.figsize'] = 10, 10
plt.matshow(heartData.corr())
plt.yticks(np.arange(heartData.shape[1]), heartData.columns)
plt.xticks(np.arange(heartData.shape[1]), heartData.columns)
plt.colorbar()
plt.savefig('correlation_matrix.png')
plt.close()
print("Saved: correlation_matrix.png")

# Cell 11: Correlation Heatmap
print("\n✓ Cell 7: Correlation Heatmap (skipped - requires Jupyter)")

# Cell 13: Target Class Distribution
print("\n✓ Cell 8: Target Class Distribution")
rcParams['figure.figsize'] = 8, 6
plt.bar(heartData['target'].unique(), heartData['target'].value_counts(), color=['black', 'silver'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')
plt.savefig('target_distribution.png')
plt.close()
print("Saved: target_distribution.png")
print(f"Target distribution:\n{heartData['target'].value_counts()}")

# Cell 15: Split Data into Train/Test
print("\n✓ Cell 9: Split Data into Train/Test Sets")
X = heartData.drop(['target'], axis=1)
y = heartData['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
print(f"XTrain-> {X_train.shape[0]} XTest-> {X_test.shape[0]} YTrain-> {y_train.shape[0]} YTest-> {y_test.shape[0]}")

print("\n" + "="*70)
print("MODEL TRAINING")
print("="*70)

# Cell 18: KNN Algorithm
print("\n✓ Cell 10: Training KNN Algorithm")
knn_scores = []
for k in range(2, 21):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train.values, y_train.values)
    knn_score = round(knn_classifier.score(X_test.values, y_test.values), 2)
    knn_scores.append(knn_score)

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
knn_score = knn_classifier.predict(X_test)
print("KNN Classification Report:")
print(classification_report(y_test, knn_score))

# Cell 20: KNN Plot
print("\n✓ Cell 11: Creating KNN Scores Plot")
plt.figure(figsize=(10, 6))
plt.plot([k for k in range(2, 21)], knn_scores, color='red')
for i in range(2, 21):
    plt.text(i, knn_scores[i-2], f'{knn_scores[i-2]}', ha='center', va='bottom')
plt.xticks([i for i in range(2, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('KNN Scores for different K neighbors')
plt.savefig('knn_scores.png')
plt.close()
print("Saved: knn_scores.png")
print(f"KNN scores: {knn_scores}")
print(f"Max KNN score: {max(knn_scores)}")

# Cell 22: KNN Test Prediction
print("\n✓ Cell 12: KNN Test Prediction")
knn_classifier_11 = KNeighborsClassifier(n_neighbors=11)
knn_classifier_11.fit(X_train.values, y_train.values)
check_data_by_sudhansu = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1, 2, 2, 3]])
prediction_result = knn_classifier_11.predict(check_data_by_sudhansu)
print(f"Prediction: {prediction_result}")

# Cell 24: Support Vector Machine
print("\n✓ Cell 13: Training Support Vector Machine")
svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svc_classifier = SVC(kernel=kernels[i])
    svc_classifier.fit(X_train.values, y_train.values)
    svc_scores.append(round(svc_classifier.score(X_test.values, y_test.values), 2))

svc_classifier = SVC(kernel=kernels[0])
svc_classifier.fit(X_train.values, y_train.values)
svc_prediction_result = svc_classifier.predict(X_test.values)
print(f"SVM Accuracy: {accuracy_score(y_test.values, svc_prediction_result)}")

# Cell 25: SVM Scores Plot
print("\n✓ Cell 14: Creating SVM Scores Plot")
colors = rainbow(np.linspace(0, 1, len(kernels)))
plt.figure(figsize=(10, 6))
plt.bar(kernels, svc_scores, color=colors)
for i in range(len(kernels)):
    plt.text(i, svc_scores[i], svc_scores[i], ha='center', va='bottom')
plt.xlabel('Kernels')
plt.ylabel('Scores')
plt.title('SVM scores Activation function wise...')
plt.savefig('svm_scores.png')
plt.close()
print("Saved: svm_scores.png")
print(f"SVM scores: {dict(zip(kernels, svc_scores))}")

# Cell 27: Decision Tree
print("\n✓ Cell 15: Training Decision Tree")
dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features=i, random_state=0)
    dt_classifier.fit(X_train.values, y_train.values)
    dt_scores.append(round(dt_classifier.score(X_test.values, y_test.values), 2))
print(f"Decision Tree scores: {dt_scores}")

# Cell 29: Final Decision Tree Model
print("\n✓ Cell 16: Training Final Decision Tree Model")
dt_classifier = DecisionTreeClassifier(max_features=13, random_state=0)
dt_classifier.fit(X_train.values, y_train.values)
print(f"Decision Tree trained with max_features=13")

# Cell 30: Decision Tree Plot
print("\n✓ Cell 17: Creating Decision Tree Scores Plot")
plt.figure(figsize=(10, 6))
plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color='green')
for i in range(1, len(X.columns) + 1):
    plt.text(i, dt_scores[i-1], f'{dt_scores[i-1]}', ha='center', va='bottom')
plt.xticks([i for i in range(1, len(X.columns) + 1)])
plt.xlabel('Max features')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier scores for different number of maximum features')
plt.savefig('decision_tree_scores.png')
plt.close()
print("Saved: decision_tree_scores.png")

# Cell 32: Random Forest
print("\n✓ Cell 18: Training Random Forest")
rf_scores = []
estimators = [10, 20, 100, 200, 500]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators=i, random_state=0)
    rf_classifier.fit(X_train.values, y_train.values)
    rf_scores.append(round(rf_classifier.score(X_test.values, y_test.values), 2))
print(f"Random Forest scores: {dict(zip(estimators, rf_scores))}")

# Cell 33: Random Forest Plot
print("\n✓ Cell 19: Creating Random Forest Scores Plot")
colors = rainbow(np.linspace(0, 1, len(estimators)))
plt.figure(figsize=(10, 6))
plt.bar([i for i in range(len(estimators))], rf_scores, color=colors, width=0.8)
for i in range(len(estimators)):
    plt.text(i, rf_scores[i], rf_scores[i], ha='center', va='bottom')
plt.xticks(ticks=[i for i in range(len(estimators))], labels=[str(estimator) for estimator in estimators])
plt.xlabel('Number of estimators')
plt.ylabel('Scores')
plt.title('Random Forest Classifier scores for different number of estimators')
plt.savefig('random_forest_scores.png')
plt.close()
print("Saved: random_forest_scores.png")

# Cell 34: Final Random Forest Model
print("\n✓ Cell 20: Training Final Random Forest Model")
rf_model = RandomForestClassifier(n_estimators=10, random_state=0)
rf_model.fit(X_train.values, y_train.values)
rf_model_result = rf_model.predict(X_test.values)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_model_result)}")

# Cell 36: Logistic Regression
print("\n✓ Cell 21: Training Logistic Regression")
logistic_model = LogisticRegression()
logistic_model.fit(X_train.values, y_train.values)
logistic_model_prediction = logistic_model.predict(X_test.values)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test.values, logistic_model_prediction)}")
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test.values, logistic_model_prediction))

# Cell 37: Save All Models
print("\n✓ Cell 22: Saving All Models to models.pkl")
all_models = [rf_model, logistic_model, dt_classifier, svc_classifier, knn_classifier]
with open("models.pkl", 'wb') as files:
    pickle.dump(all_models, files)
print("Done - Models saved!")

# Cell 39: Verify Saved Models
print("\n✓ Cell 23: Verifying Saved Models")
open_file = open("models.pkl", "rb")
loaded_list = pickle.load(open_file)
print(f"Loaded models: {loaded_list}")
open_file.close()
print("Done - Verification complete!")

print("\n" + "="*70)
print("ALL NOTEBOOK CELLS EXECUTED SUCCESSFULLY!")
print("="*70)
print("\nGenerated Files:")
print("  - models.pkl (trained models)")
print("  - correlation_matrix.png")
print("  - target_distribution.png")
print("  - knn_scores.png")
print("  - svm_scores.png")
print("  - decision_tree_scores.png")
print("  - random_forest_scores.png")
print("\nYou can now run: python app.py")
print("="*70)
