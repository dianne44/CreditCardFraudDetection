import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import shap
import xgboost as xgb
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datasets import load_dataset
from imblearn.over_sampling import RandomOverSampler

# Step 1: Load Dataset
ds = load_dataset("thomask1018/credit_card_fraud")
data = ds['train'].to_pandas()

# Step 2: Data Preprocessing
X = data.drop(columns=['Class'])
y = data['Class']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Balance the Dataset using Oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_scaled, y)

# Step 4: Feature Selection using SHAP
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_resampled, y_resampled)
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_resampled)
shap_importance = np.abs(shap_values.values).mean(axis=0)
top_features = np.argsort(shap_importance)[-10:]
X_reduced = X.iloc[:, top_features]

# Re-scale reduced features
X_scaled_reduced = scaler.fit_transform(X_reduced)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled_reduced, y_resampled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_train_torch = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test_torch = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# Step 6: Define Quantum Device
n_qubits = X_train.shape[1]
dev = qml.device("default.qubit", wires=n_qubits)

# Step 7: Define Quantum Circuit
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(features=inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# Step 8: Define Quantum Neural Network (QNN)
class QuantumFraudDetector(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qnode = qml.qnn.TorchLayer(quantum_circuit, self.weight_shapes)

    def forward(self, x):
        return torch.sigmoid(self.qnode(x))

# Step 9: Initialize Quantum Model
n_layers = 3
model = QuantumFraudDetector(n_qubits, n_layers)

# Step 10: Define Loss and Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

# Step 11: Train Quantum Model
epochs = 10
batch_size = 32
dataset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

start_time = time.time()  # Start timing

for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        y_pred = model(batch_X).view(-1)
        loss = loss_fn(y_pred, batch_y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

training_time = time.time() - start_time  # End timing

# Step 12: Evaluate Model on Test Data
with torch.no_grad():
    y_train_pred = model(X_train_torch).view(-1)
    y_train_pred = (y_train_pred > 0.5).float()
    
    y_test_pred = model(X_test_torch).view(-1)
    y_test_pred = (y_test_pred > 0.5).float()

# Compute Training & Testing Accuracy
train_accuracy = accuracy_score(y_train_torch, y_train_pred)
test_accuracy = accuracy_score(y_test_torch, y_test_pred)

# Compute Precision, Recall, F1-Score, and Specificity
precision = precision_score(y_test_torch, y_test_pred)
recall = recall_score(y_test_torch, y_test_pred)
f1 = f1_score(y_test_torch, y_test_pred)

# Compute Specificity
tn, fp, fn, tp = confusion_matrix(y_test_torch, y_test_pred).ravel()
specificity = tn / (tn + fp)

# Step 13: Print Results
print("\n--- Quantum Model Performance Metrics ---")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Specificity: {specificity:.2f}")

# Step 14: Display Sample Predictions
print("\nSample Predictions:")
for i in range(5):
    print(f"Actual: {int(y_test_torch[i].item())}, Predicted: {int(y_test_pred[i].item())}")

print("\n✅ Training & evaluation completed!")
