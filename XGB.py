# 📌 Step 1: Load the dataset
import pandas as pd
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 📌 Step 2: Load dataset and select relevant features
dataset = load_dataset("thomask1018/credit_card_fraud")
df = pd.DataFrame(dataset['train'])  # Convert dataset to DataFrame

feature_cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 
                'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
df = df[[col for col in feature_cols if col in df.columns] + ['Class']]

# 📌 Step 3: Normalize the 'Amount' feature using MinMaxScaler
scaler = MinMaxScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# 📌 Step 4: Split data into features (X) and target variable (y)
X = df.drop(columns=['Class'])
y = df['Class']

# 📌 Step 5: Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 📌 Step 6: Train XGBoost Model with specified learning rate
learning_rate = 0.1  # Set learning rate
xgb_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, learning_rate=learning_rate)
xgb_model.fit(X_train, y_train)

# 📌 Step 7: Make Predictions with XGBoost
y_pred_xgb = xgb_model.predict(X_test)

# 📌 Step 8: Evaluate XGBoost Model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

# 📌 Step 9: Compute Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
tn, fp, fn, tp = cm_xgb.ravel()
specificity_xgb = tn / (tn + fp)
sensitivity_xgb = tp / (tp + fn)

# 📌 Step 10: Print XGBoost Model Performance and Learning Rate
print("\n🔍 XGBoost Model Performance:")
print(f"✅ Accuracy:     {accuracy_xgb * 100:.2f}%")
print(f"✅ Precision:    {precision_xgb:.4f}")
print(f"✅ Recall:       {recall_xgb:.4f}")
print(f"✅ F1-Score:     {f1_xgb:.4f}")
print(f"✅ Specificity:  {specificity_xgb:.4f}")
print(f"✅ Sensitivity:  {sensitivity_xgb:.4f}")
print(f"✅ Learning Rate: {learning_rate}")

# 📌 Step 11: Print Confusion Matrix as a Table
cm_df = pd.DataFrame(cm_xgb, 
                     index=["Actual: No Fraud (0)", "Actual: Fraud (1)"], 
                     columns=["Predicted: No Fraud (0)", "Predicted: Fraud (1)"])
print("\n📊 Confusion Matrix (Table Format):")
print(cm_df)

# 📌 Step 12: Plot Confusion Matrix using Seaborn
plt.figure(figsize=(6, 5))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["No Fraud (0)", "Fraud (1)"], 
            yticklabels=["No Fraud (0)", "Fraud (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("🔍 Confusion Matrix for XGBoost Model")
plt.show()
