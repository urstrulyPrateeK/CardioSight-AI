import pandas as pd
import numpy as np
import pickle
import os
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Concatenate, Flatten

# --- Configuration ---
DATA_PATH = 'dataset/heart_data_set.csv'
MODEL_DIR = 'cardiosight_models'
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Data Preparation ---
print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Basic preprocessing
X = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y = df['target']

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Save scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

# Split Main Train/Test (Hold out test set for final eval)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Data Loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- 2. Base Classifiers (Stage 1) ---
print("Training Base Classifiers...")

# Models as per paper
models = {
    'LR': LogisticRegression(),
    'KNN': KNeighborsClassifier(n_neighbors=11), # Paper mentions K=11 was stable
    'SVM': SVC(probability=True, kernel='linear'), # Paper mentions linear kernel was best
   'RF': RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_split=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
),
    'DT': DecisionTreeClassifier(max_features=11) # Paper mentions max_features=11
}

# We need OOF (Out-Of-Fold) predictions for the train set to train the Meta-Learner without leakage.
# And we need simple predictions for the test set.

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Arrays to store meta-features (probabilities from base models)
meta_X_train = np.zeros((X_train.shape[0], len(models)))
meta_X_test = np.zeros((X_test.shape[0], len(models)))

trained_models = {}

for i, (name, model) in enumerate(models.items()):
    print(f"Training {name}...")
    
    # 1. Generate OOF predictions for training data
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        
        # Clone model for fold to ensure fresh training
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_fold_train, y_fold_train)
        
        # Predict probabilities (taking prob of class 1)
        meta_X_train[val_idx, i] = fold_model.predict_proba(X_fold_val)[:, 1]

    # 2. Train on full training set and predict on test set
    full_model = model
    full_model.fit(X_train, y_train)
    meta_X_test[:, i] = full_model.predict_proba(X_test)[:, 1]
    
    # Save the full model
    joblib.dump(full_model, os.path.join(MODEL_DIR, f'{name}.pkl'))
    trained_models[name] = full_model
    
    print(f"{name} Test Accuracy: {accuracy_score(y_test, full_model.predict(X_test)):.4f}")

print("Base models trained and saved.")

# # --- 3. Prepare Inputs for CNN-LSTM (Stage 2) ---
# # The paper mentions concatenating:
# # 1. Original features (13)
# # 2. Base model probabilities (5)
# # 3. Pseudo-signal windows for CNN/LSTM

# # Combine features for Dense input
# X_train_dense = np.hstack([X_train, meta_X_train])
# X_test_dense = np.hstack([X_test, meta_X_test])

# # Construct Pseudo-Signal for CNN/LSTM
# # Paper: "stacking time varying fields like max heart rate and ST depression"
# # We'll treat the feature vector itself as a sequence or select specific subset.
# # Let's take the whole feature vector (size 13) and reshape it to (13, 1) or keep it meant for 1D Conv.
# # Actually, CNN 1D expects (batch, steps, channels).
# # Let's treat the 13 features as a time-series of length 13, 1 channel.
# X_train_seq = X_train.reshape(X_train.shape[0], 13, 1)
# X_test_seq = X_test.reshape(X_test.shape[0], 13, 1)

# print(f"Meta-Learner Input Shapes: Dense {X_train_dense.shape}, Seq {X_train_seq.shape}")

# # --- 4. Build CNN-LSTM Model ---
# def build_cnn_lstm_model(input_shape_seq, input_shape_dense):
#     # Branch 1: CNN-LSTM
#     seq_input = Input(shape=input_shape_seq, name='seq_input')
    
#     # CNN Block 1
#     x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(seq_input)
#     x = MaxPooling1D(pool_size=2)(x)
    
#     # CNN Block 2
#     x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
#     x = MaxPooling1D(pool_size=2)(x)
    
#     # LSTM Layer
#     x = LSTM(64)(x)
    
#     # Branch 2: Dense Features (Original + Meta)
#     dense_input = Input(shape=input_shape_dense, name='dense_input')
    
#     # Concatenate
#     concat = Concatenate()([x, dense_input])
    
#     # Fully Connected Layers
#     fc = Dense(128, activation='relu')(concat)
#     fc = Dropout(0.3)(fc)
#     fc = Dense(64, activation='relu')(fc)
#     fc = Dropout(0.3)(fc)
    
#     output = Dense(1, activation='sigmoid')(fc)
    
#     model = Model(inputs=[seq_input, dense_input], outputs=output)
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# cnn_lstm_model = build_cnn_lstm_model((13, 1), (18,))
# # print(cnn_lstm_model.summary())

# # --- 5. Train Meta-Learner ---
# print("Training CNN-LSTM Meta-Learner...")
# history = cnn_lstm_model.fit(
#     [X_train_seq, X_train_dense], y_train,
#     epochs=20, # Paper mentions 20-30 epochs
#     batch_size=32,
#     validation_split=0.2,
#     verbose=1
# )

# # --- 6. Evaluation ---
# print("Evaluating on Test Set...")
# loss, acc = cnn_lstm_model.evaluate([X_test_seq, X_test_dense], y_test)
# print(f"Final CNN-LSTM Test Accuracy: {acc:.4f}")

# Save the Keras model
# cnn_lstm_model.save(os.path.join(MODEL_DIR, 'cnn_lstm.h5'))
# print(f"All models saved to {MODEL_DIR}")
# print("Training complete. Using base models only.")
print("All base models saved successfully in cardiosight_models folder.")
