import pandas as pd
import numpy as np
import torch
import pennylane as qml
import shap
import xgboost as xgb
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

# Step 1: Load Dataset
ds = load_dataset("thomask1018/credit_card_fraud")
data = ds['train'].to_pandas()

# Step 2: Data Preprocessing
X = data.drop(columns=['Class'])
y = data['Class']

# Split before scaling (prevents data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Balance the Training Dataset using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Step 4: Feature Selection using SHAP
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train_resampled, y_train_resampled)

explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_train_resampled)
shap_importance = np.abs(shap_values.values).mean(axis=0)
top_features = np.argsort(shap_importance)[-28:]

# Keep only top features
X_train_selected = X_train_resampled[:, top_features]
X_test_selected = X_test_scaled[:, top_features]

# Step 5: Reduce Dataset Size for Feasibility
subset_size = min(5000, len(X_train_selected))  # Ensure subset doesn't exceed available data
X_train_final, _, y_train_final, _ = train_test_split(X_train_selected, y_train_resampled, train_size=subset_size, random_state=42, stratify=y_train_resampled)

# Step 6: Define Quantum Device
n_qubits = X_train_final.shape[1]
dev = qml.device("default.qubit", wires=n_qubits)

# Step 7: Define Quantum Kernel
@qml.qnode(dev)
def quantum_kernel(x1, x2):
    qml.templates.AngleEmbedding(x1, wires=range(n_qubits))
    qml.templates.AngleEmbedding(x2, wires=range(n_qubits))
    return qml.expval(qml.Hermitian(np.outer(x1, x2), wires=range(n_qubits)))

# Step 8: Compute Quantum Kernel Approximation
nystroem = Nystroem(kernel='rbf', n_components=500)  # Reduced from 3000 to 500
X_train_transformed = nystroem.fit_transform(X_train_final)
X_test_transformed = nystroem.transform(X_test_selected)

# Step 9: Train SVM with Quantum Kernel Approximation
start_time = time.time()
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train_transformed, y_train_final)
training_time = time.time() - start_time

# Step 10: Evaluate Model on Test Data
y_train_pred = svm.predict(X_train_transformed)
y_test_pred = svm.predict(X_test_transformed)
y_test_probs = svm.predict_proba(X_test_transformed)[:, 1]  # Probabilities for AUC metrics

# Performance Metrics
train_accuracy = accuracy_score(y_train_final, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, y_test_probs)
pr_auc = average_precision_score(y_test, y_test_probs)
tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
specificity = tn / (tn + fp)

# Print Results
print(f"Training Time: {training_time:.2f} seconds")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")
print(f"Precision-Recall AUC: {pr_auc:.2f}")
print(f"Specificity: {specificity:.2f}")

# Step 11: Display Sample Predictions
print("\nSample Predictions:")
for i in range(5):
    print(f"Actual: {int(y_test.iloc[i])}, Predicted: {int(y_test_pred[i])}")
