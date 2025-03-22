import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
print("Loading dataset...")
dataset = load_dataset("thomask1018/credit_card_fraud")
df = dataset['train'].to_pandas()

# Preprocess features and labels
features = df.drop(columns=["Class"]).values
labels = df["Class"].values

# Normalize features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Reduce dimensions using PCA (now with 7 components)
num_qubits = 7  # Updated number of qubits to match the number of features
pca = PCA(n_components=num_qubits)
features = pca.fit_transform(features)

# Convert to PyTorch tensors
tensor_features = torch.tensor(features, dtype=torch.float32)
tensor_labels = torch.tensor(labels, dtype=torch.int64)

# Quantum device setup
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(num_qubits))  # Match feature count
    qml.BasicEntanglerLayers(weights, wires=range(num_qubits))
    return qml.expval(qml.PauliZ(0))

# Hybrid Quantum Neural Network
class QuantumQNetwork(nn.Module):
    def __init__(self):
        super(QuantumQNetwork, self).__init__()
        weight_shapes = {"weights": (3, num_qubits)}  # Match qubit count
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.fc = nn.Linear(1, 2)  # Output layer for Fraud/Not Fraud

    def forward(self, x):
        q_out = self.q_layer(x)
        return self.fc(q_out.unsqueeze(1))  # Ensure correct shape

# Initialize network, loss, and optimizer
q_network = QuantumQNetwork()
optimizer = optim.Adam(q_network.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training parameters
num_episodes = 100
batch_size = 32

def train_qdqn(features, labels):
    for episode in range(num_episodes):
        optimizer.zero_grad()
        
        batch_indices = random.sample(range(len(features)), batch_size)
        batch_features = tensor_features[batch_indices]
        batch_labels = tensor_labels[batch_indices]
        
        q_values = q_network(batch_features)
        target_q_values = torch.zeros_like(q_values)
        
        for i in range(batch_size):
            target_q_values[i, batch_labels[i]] = 1.0  # Reward-based Q-value update
        
        loss = loss_fn(q_values, target_q_values)
        loss.backward()
        optimizer.step()
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Loss: {loss.item():.4f}")

# Train the Quantum DQN
train_qdqn(tensor_features, tensor_labels)

# Evaluation
with torch.no_grad():
    predictions = torch.argmax(q_network(tensor_features), dim=1).numpy()
    true_labels = tensor_labels.numpy()

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()

    # Compute metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)  # Sensitivity
    specificity = tn / (tn + fp)
    f1 = f1_score(true_labels, predictions)
    effort_score = (fp + fn) / len(true_labels)

    # Print results
    print(f"\nFinal Model Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Sensitivity (Recall): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Effort Score: {effort_score:.4f}")
