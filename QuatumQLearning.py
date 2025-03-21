import pennylane as qml
import numpy as np
import random
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset from Hugging Face
dataset = load_dataset("thomask1018/credit_card_fraud")
df = dataset['train'].to_pandas()

# Extract features and labels
features = df.drop(columns=["Class"]).values  # Transaction features (30 columns)
labels = df["Class"].values  # Fraud labels (0 = legit, 1 = fraud)

# Normalize features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Reduce feature dimension to 10 using PCA
pca = PCA(n_components=10)
features = pca.fit_transform(features)

# Quantum Q-learning Parameters
num_qubits = 10  # Match the 10 selected features
dev = qml.device("default.qubit", wires=num_qubits)

# Quantum Circuit for Q-value Estimation
@qml.qnode(dev)
def quantum_q_value(state, action, weights):
    """Quantum Q-value estimation"""
    qml.AngleEmbedding(state, wires=range(num_qubits))  # Encode state into quantum circuit

    # Ensure weights are 2D (1 layer, num_qubits)
    weights = np.reshape(weights, (1, num_qubits))

    qml.templates.BasicEntanglerLayers(weights, wires=range(num_qubits))  # Fix weight issue

    return qml.expval(qml.PauliZ(0))  # Q-value estimate

# Quantum Q-Learning Class
class QuantumQLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # Quantum Q-table

    def get_q_value(self, state, action):
        """Retrieve Q-value from the quantum circuit"""
        key = tuple(np.round(state, 5)) + (action,)
        if key not in self.q_table:
            self.q_table[key] = np.random.randn(1, num_qubits)  # Initialize correctly
        return quantum_q_value(state, action, self.q_table[key].flatten())  # Flatten for function call

    def update_q_value(self, state, action, reward, next_state):
        """Quantum Q-learning update rule"""
        key = tuple(np.round(state, 5)) + (action,)  # Round floats for stable keys
        next_key = [tuple(np.round(next_state, 5)) + (a,) for a in [0, 1]]

        # Initialize missing keys
        for nk in next_key:
            if nk not in self.q_table:
                self.q_table[nk] = np.random.randn(1, num_qubits)

        next_q_value = max([self.get_q_value(next_state, a) for a in [0, 1]])

        if key not in self.q_table:
            self.q_table[key] = np.random.randn(1, num_qubits)  # Initialize missing Q-values

        self.q_table[key] += self.alpha * (reward + self.gamma * next_q_value - self.q_table[key])

    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        key = tuple(np.round(state, 5))  # Round floats for stable keys
        if np.random.rand() < self.epsilon:
            return random.choice([0, 1])  # Explore (random choice)
        return max([0, 1], key=lambda a: self.get_q_value(state, a))  # Exploit (choose best Q-value)

# Initialize Quantum Q-Learning Agent
agent = QuantumQLearningAgent()

# Train the Agent on Fraud Detection Data
def train_agent(features, labels, episodes=500):
    dataset = list(zip(features, labels))  # Combine features with labels
    for episode in range(episodes):
        state, true_label = random.choice(dataset)  # Select a transaction sample
        action = agent.choose_action(state)
        reward = 1 if action == true_label else -1  # Reward based on correct classification
        next_state, _ = random.choice(dataset)  # Sample next state
        agent.update_q_value(state, action, reward, next_state)

# Train Model
train_agent(features, labels, episodes=1000)

# Evaluate Model
def evaluate_agent(features, labels):
    y_true, y_pred = [], []

    for state, true_label in zip(features, labels):
        action = agent.choose_action(state)
        y_true.append(true_label)
        y_pred.append(action)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)  # Sensitivity
    f1 = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)  # Specificity calculation

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")

    return accuracy, precision, recall, f1, specificity

# Run evaluation
evaluate_agent(features, labels)
