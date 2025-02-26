import numpy as np
import pandas as pd
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter

# Step 1: Load the dataset
dataset = load_dataset("thomask1018/credit_card_fraud")
df = dataset['train'].to_pandas()

# Step 2: Data Preprocessing
df = df.drop(columns=["Transaction_ID"], errors='ignore')  # Remove unnecessary column
X = df.drop(columns=["Class"])  # Features
y = df["Class"].values  # Target (Fraud: 1, Not Fraud: 0)

# Normalize numerical features using Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Define the Quantum Device
n_qubits_per_batch = 3  # Process 3 features at a time
dev = qml.device("default.qubit", wires=n_qubits_per_batch)

@qml.qnode(dev)
def qpca_quantum_embedding(x):
    """Quantum feature extraction using QPCA"""
    qml.AngleEmbedding(x, wires=range(n_qubits_per_batch))
    qml.templates.BasicEntanglerLayers(weights=np.ones((3, n_qubits_per_batch)), wires=range(n_qubits_per_batch))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits_per_batch)]

def apply_qpca_batch(X, batch_size=3):
    """Process features in batches"""
    n_features = X.shape[1]
    batched_features = []

    for i in range(0, n_features, batch_size):
        batch = X[:, i:i + batch_size]
        if batch.shape[1] < batch_size:
            pad_width = batch_size - batch.shape[1]
            batch = np.pad(batch, ((0, 0), (0, pad_width)), mode='constant')
        
        batch_qpca = np.array([qpca_quantum_embedding(sample) for sample in batch])
        batched_features.append(batch_qpca)

    return np.hstack(batched_features)

# Step 4: Apply QPCA Feature Extraction
X_qpca = apply_qpca_batch(X_scaled[:5000])
y_qpca = y[:5000]

# Step 5: Split Data into Training & Testing
X_train, X_test, y_train, y_test = train_test_split(X_qpca, y_qpca, test_size=0.2, random_state=42)

# Step 6: Balance Data Using SMOTE or RandomOverSampler
print("Class distribution before resampling:", Counter(y_train))

if len(np.unique(y_train)) > 1:  # Ensure at least two classes exist
    min_class_samples = np.bincount(y_train).min()
    k_neighbors = min(2, min_class_samples - 1)  # Adjust based on class size
    if k_neighbors > 0:
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    else:
        print("⚠️ Not enough fraud samples for SMOTE. Using RandomOverSampler instead.")
        ros = RandomOverSampler(random_state=42)
        X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)
else:
    print("⚠️ Only one class present, resampling skipped.")
    X_train_balanced, y_train_balanced = X_train, y_train

print("Class distribution after resampling:", Counter(y_train_balanced))

# Step 7: Train Decision Tree Classifier (Teacher)
teacher_model = DecisionTreeClassifier(max_depth=5, random_state=42)
teacher_model.fit(X_train_balanced, y_train_balanced)

# Generate soft labels from Decision Tree
y_train_teacher = teacher_model.predict(X_train_balanced)
y_test_teacher = teacher_model.predict(X_test)

# Step 8: Train K-NN Classifier (Student)
if len(np.unique(y_train_teacher)) > 1:
    student_knn = KNeighborsClassifier(n_neighbors=3)
    student_knn.fit(X_train_balanced, y_train_teacher)

    # Step 9: Make Predictions with Student Model
    y_pred_student = student_knn.predict(X_test)

    # Step 10: Evaluate the Student Model
    accuracy = accuracy_score(y_test_teacher, y_pred_student)
    precision = precision_score(y_test_teacher, y_pred_student, zero_division=0)
    recall = recall_score(y_test_teacher, y_pred_student, zero_division=0)
    f1 = f1_score(y_test_teacher, y_pred_student, zero_division=0)
    
    # Step 11: Compute Confusion Matrix
    cm = confusion_matrix(y_test_teacher, y_pred_student)
    cm_df = pd.DataFrame(cm, 
                         index=["Actual: No Fraud (0)", "Actual: Fraud (1)"], 
                         columns=["Predicted: No Fraud (0)", "Predicted: Fraud (1)"])
    print("\n📊 Confusion Matrix:")
    print(cm_df)

    # Step 12: Print Evaluation Metrics
    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [accuracy, precision, recall, f1]
    })
    print("\n📊 Evaluation Metrics QKNN:")
    print(metrics.to_string(index=False))

    # Step 13: Visualize Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Fraud (0)", "Fraud (1)"], yticklabels=["No Fraud (0)", "Fraud (1)"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("🔍 Confusion Matrix for K-NN Student Model")
    plt.show()
else:
    print("⚠️ Not enough classes in teacher labels. K-NN cannot be trained.")
