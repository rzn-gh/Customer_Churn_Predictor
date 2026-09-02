import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load Data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Data Cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce').fillna(0)
df = df.drop(columns=['customerID'])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 3. Industry Feature Engineering
# Feature A: Avg Cost per Month Ratio
df['AvgCostPerMonth'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], df['MonthlyCharges'])

# Feature B: Total Subscribed Services Count
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['TotalServices'] = (df[service_cols] == 'Yes').sum(axis=1)

# Feature C: Has Tech Protection (Security or Tech Support)
df['HasProtection'] = ((df['OnlineSecurity'] == 'Yes') | (df['TechSupport'] == 'Yes')).astype(int)

# 4. Feature Encoding
encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 5. Split Data
X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train LightGBM with Class Balance Handling
model = LGBMClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    class_weight='balanced', # Handles imbalanced churn classes
    random_state=42
)
model.fit(X_train, y_train)

# 8. Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Save Artifacts
joblib.dump(model, 'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')
print("Pro Model training complete and saved!")