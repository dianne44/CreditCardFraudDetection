# Import necessary libraries
import pandas as pd
from datasets import load_dataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load the dataset from Hugging Face
dataset = load_dataset("thomask1018/credit_card_fraud")

# Convert dataset to pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Define feature columns
feature_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Select features and target variable
df = df[[col for col in feature_cols if col in df.columns] + ['Class']]

# Normalize 'Amount' feature
scaler = MinMaxScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# Split into features and labels
X = df.drop(columns=['Class'])
y = df['Class']

# Split data into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Train Decision Tree Model (Teacher)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Generate probability labels (soft labels)
y_train_proba = dt_model.predict_proba(X_train)[:, 1]  # Probabilities of being fraud
y_test_proba = dt_model.predict_proba(X_test)[:, 1]

# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_train.values, dtype=torch.float32)
X_test_torch = torch.tensor(X_test.values, dtype=torch.float32)
y_train_proba_torch = torch.tensor(y_train_proba, dtype=torch.float32).reshape(-1, 1)
y_test_proba_torch = torch.tensor(y_test_proba, dtype=torch.float32).reshape(-1, 1)
# Define feature batches (each batch has 5 features)
batch_size = 5
num_batches = X_train.shape[1] // batch_size  # 30 features -> 6 batches

feature_batches = [X_train.columns[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
# Define Quantum Device
num_qubits = batch_size  # Each batch has 5 features
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(num_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    return qml.expval(qml.PauliZ(0))

# Quantum Student Model
class QuantumStudent(nn.Module):
    def __init__(self, num_qubits, n_layers=3):
        super().__init__()
        self.weight_shapes = {"weights": (n_layers, num_qubits, 3)}
        self.qnode = qml.qnn.TorchLayer(quantum_circuit, self.weight_shapes)

    def forward(self, x):
        return torch.sigmoid(self.qnode(x))

# Initialize multiple QVC models for each batch
qvc_models = [QuantumStudent(num_qubits) for _ in range(num_batches)]
optimizers = [optim.Adam(model.parameters(), lr=0.01) for model in qvc_models]
epochs = 10
batch_size = 32
loss_fn = nn.BCELoss()

# Training loop for each QVC student
for i, (features, model, optimizer) in enumerate(zip(feature_batches, qvc_models, optimizers)):
    print(f"\n🔬 Training QVC Model {i+1} on Features: {features}")

    # Select feature subset
    X_train_batch = X_train[features]
    X_train_torch_batch = torch.tensor(X_train_batch.values, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_train_torch_batch, y_train_proba_torch)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
# Aggregate all QVC predictions
qvc_preds = torch.stack([model(X_test_torch[:, i * batch_size:(i + 1) * batch_size]).view(-1) for i, model in enumerate(qvc_models)], dim=1)
final_preds = torch.mean(qvc_preds, dim=1) > 0.5  # Average and threshold

# Evaluate the hybrid model
accuracy = accuracy_score(y_test, final_preds)
print(f"\n🏆 Final Hybrid Model Accuracy: {accuracy * 100:.2f}%")
