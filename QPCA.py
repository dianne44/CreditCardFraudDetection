import numpy as np
import pandas as pd
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import Parallel, delayed

# Step 1: Load the dataset
dataset = load_dataset("thomask1018/credit_card_fraud")
df = dataset['train'].to_pandas()

# Step 2: Data Preprocessing
df = df.drop(columns=["Transaction_ID"], errors='ignore')  # Remove unnecessary column
X = df.drop(columns=["Class"])  # Features
y = df["Class"]  # Target (Fraud: 1, Not Fraud: 0)

# Normalize numerical features using Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Define the Quantum Device (Batch Processing)
n_qubits_per_batch = 3  # Process 3 features at a time for efficiency
dev = qml.device("default.qubit", wires=n_qubits_per_batch)

@qml.qnode(dev)
def qpca_quantum_embedding(x):
    """Quantum feature extraction using QPCA on small batches"""
    qml.AngleEmbedding(x, wires=range(n_qubits_per_batch))
    qml.templates.BasicEntanglerLayers(weights=np.ones((3, n_qubits_per_batch)), wires=range(n_qubits_per_batch))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits_per_batch)]

def apply_qpca_batch(X, batch_size=3):
    """Process features in batches to reduce execution time"""
    n_features = X.shape[1]
    batched_features = []

    for i in range(0, n_features, batch_size):
        batch = X[:, i:i + batch_size]  # Select batch features
        if batch.shape[1] < batch_size:  # Pad if necessary
            pad_width = batch_size - batch.shape[1]
            batch = np.pad(batch, ((0, 0), (0, pad_width)), mode='constant')
        
        batch_qpca = np.array([qpca_quantum_embedding(sample) for sample in batch])
        batched_features.append(batch_qpca)

    return np.hstack(batched_features)  # Concatenate results

# Step 4: Apply QPCA Feature Extraction with Parallel Processing
def parallel_qpca(X, batch_size=3, n_jobs=-1):
    """Parallel batch processing for faster execution"""
    n_features = X.shape[1]
    results = Parallel(n_jobs=n_jobs)(
        delayed(lambda batch: np.array([qpca_quantum_embedding(sample) for sample in batch]))(
            X[:, i:i + batch_size]
        ) for i in range(0, n_features, batch_size)
    )
    return np.hstack(results)

# Reduce dataset size for efficient quantum processing
X_qpca = parallel_qpca(X_scaled[:5000], batch_size=3)
y_qpca = y[:5000].values

# Step 5: Split Data into Training & Testing
X_train, X_test, y_train, y_test = train_test_split(X_qpca, y_qpca, test_size=0.2, random_state=42)

# Step 6: Train K-NN Classifier using QPCA Features
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = knn.predict(X_test)

# Step 8: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Compute specificity & sensitivity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

# Define Effort Score (custom metric)
effort_score = (precision + recall) / 2  

# Compute Learning Rate (Example: 1 / Iterations)
iterations = 100  # Hypothetical number of iterations
learning_rate = 1 / iterations

# Step 9: Print Evaluation Metrics as a Table
metrics_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "Sensitivity", "Effort Score", "Learning Rate"],
    "Value": [accuracy, precision, recall, f1, specificity, sensitivity, effort_score, learning_rate]
})

print("\n📊 **Evaluation Metrics Table:**")
print(metrics_table.to_string(index=False))

# Step 10: Print Confusion Matrix as a Table
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, 
                     index=["Actual: No Fraud (0)", "Actual: Fraud (1)"], 
                     columns=["Predicted: No Fraud (0)", "Predicted: Fraud (1)"])

print("\n📊 **Confusion Matrix Table:**")
print(cm_df)

# Step 11: Visualize Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["No Fraud (0)", "Fraud (1)"], 
            yticklabels=["No Fraud (0)", "Fraud (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("🔍 Confusion Matrix for K-NN Model (QPCA)")
plt.show()

# Step 12: Plot Learning Rate vs Evaluation Metrics
plt.figure(figsize=(8, 5))
metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "Sensitivity", "Effort Score"]
metrics_values = [accuracy, precision, recall, f1, specificity, sensitivity, effort_score]

plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'brown'])
plt.axhline(y=learning_rate, color='black', linestyle='--', label=f"Learning Rate ({learning_rate:.4f})")
plt.xticks(rotation=45)
plt.ylabel("Metric Values")
plt.title("📊 Evaluation Metrics vs. Learning Rate")
plt.legend()
plt.show()
