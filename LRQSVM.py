
import time
import pennylane as qml
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
from sklearn.svm import SVC
from imblearn.under_sampling import RandomUnderSampler

# Load dataset
data = pd.read_csv('/content/drive/MyDrive/creditcard.csv')



# Define feature matrix (X) and target variable (y)
data = data.dropna(subset=['Class'])
X = data.drop(['Class', 'Time'], axis=1)  # Exclude 'Time'
selected_features = X.columns[:10]  # Select the first 10 numerical features
X = X[selected_features]  # Keep only 10 features
y = data['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Undersample to balance class distribution
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_scaled, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Define Quantum Device
dev = qml.device("lightning.qubit", wires=10)  # Match number of features

# Quantum Kernel Function
def quantum_kernel(inputs1, inputs2):
    @qml.qnode(dev)
    def circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=range(len(x1)))
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(len(x2)))
        return qml.probs(wires=range(len(x1)))

    probs = circuit(inputs1, inputs2)
    return probs[0]

# Compute Quantum Kernel Matrix in Batches
def compute_batch(start_idx, end_idx, X_data1, X_data2):
    return np.array([[quantum_kernel(X_data1[i], X_data2[j]) for j in range(len(X_data2))]
                     for i in range(start_idx, end_idx)])

def compute_kernel_matrix_parallel(X_data1, X_data2, batch_size=50):
    total_rows = len(X_data1)
    kernel_matrix = np.zeros((total_rows, len(X_data2)))

    with multiprocessing.Pool() as pool:
        results = pool.starmap(compute_batch, [(i, min(i + batch_size, total_rows), X_data1, X_data2)
                                               for i in range(0, total_rows, batch_size)])

    return np.vstack(results)

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1)
log_reg.fit(X_train, y_train)

# Train Quantum SVM
print("Computing Quantum Kernel Matrix (Training)...")
kernel_matrix_train = compute_kernel_matrix_parallel(X_train, X_train)

print("Computing Quantum Kernel Matrix (Testing)...")
kernel_matrix_test = compute_kernel_matrix_parallel(X_test, X_train)

qsvm = SVC(kernel='precomputed')
qsvm.fit(kernel_matrix_train, y_train)

# Predict using Logistic Regression
y_pred_log_reg = log_reg.predict(X_test)

# Predict using Quantum SVM
y_pred_qsvm = qsvm.predict(kernel_matrix_test)

# Hybrid Model Predictions (Weighted Average)
hybrid_preds = (0.5 * y_pred_log_reg + 0.5 * y_pred_qsvm).round().astype(int)

# Evaluation Metrics
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    effort_score = (precision + recall) / 2  # Custom metric

    print(f"\n📊 {model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Effort Score: {effort_score:.4f}")

    return accuracy, precision, recall, f1, specificity, effort_score

# Evaluate Models
metrics_log_reg = evaluate_model(y_test, y_pred_log_reg, "Logistic Regression")
metrics_qsvm = evaluate_model(y_test, y_pred_qsvm, "Quantum SVM")
metrics_hybrid = evaluate_model(y_test, hybrid_preds, "Hybrid Model")

# Confusion Matrices
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

plot_confusion_matrix(y_test, y_pred_log_reg, "Logistic Regression")
plot_confusion_matrix(y_test, y_pred_qsvm, "Quantum SVM")
plot_confusion_matrix(y_test, hybrid_preds, "Hybrid Model")

# ROC Curves
def plot_roc_curve(y_true, y_pred_probs, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

plt.figure(figsize=(8, 6))
plot_roc_curve(y_test, y_pred_log_reg, "Logistic Regression")
plot_roc_curve(y_test, y_pred_qsvm, "Quantum SVM")
plot_roc_curve(y_test, hybrid_preds, "Hybrid Model")
plt.show()

# Learning Rate vs Model Performance
learning_rates = np.linspace(0.01, 1.0, 10)
performances = {"Accuracy": [], "Precision": [], "Recall": [], "F1 Score": []}

for lr in learning_rates:
    hybrid_preds = (lr * y_pred_log_reg + (1 - lr) * y_pred_qsvm).round().astype(int)
    acc, prec, rec, f1, _, _ = evaluate_model(y_test, hybrid_preds, f"Hybrid Model (LR={lr:.2f})")

    performances["Accuracy"].append(acc)
    performances["Precision"].append(prec)
    performances["Recall"].append(rec)
    performances["F1 Score"].append(f1)

plt.figure(figsize=(8, 6))
for metric, values in performances.items():
    plt.plot(learning_rates, values, label=metric)

plt.xlabel("Learning Rate")
plt.ylabel("Performance")
plt.title("Learning Rate vs Model Performance")
plt.legend()
plt.show()
