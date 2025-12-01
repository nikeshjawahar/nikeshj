<a href="https://colab.research.google.com/github/dharanesh-1/dharanesh-1/blob/main/Copy_of_Untitled5.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.impute import SimpleImputer
df = pd.read_csv("creditcard.csv")

print("✔ Dataset Loaded Successfully!\n")
print(df.head())
df = pd.read_csv("creditcard.csv")

print("✔ Dataset Loaded Successfully!\n")

# The last column is assumed to be the target variable (fraud column)
y = df.iloc[:, -1]
X = df.iloc[:, :-1]

numeric_cols = X.select_dtypes(include=['int64','float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('passthrough', 'passthrough')
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)
print("✔ Preprocessing pipeline defined with imputation!")
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Missing values in X before imputation:")
print(X.isnull().sum())
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# FRAUD PROBABILITY HISTOGRAM
plt.figure(figsize=(6,4))
plt.hist(y_proba, bins=20, edgecolor="black")
plt.title("Fraud Probability Distribution")
plt.xlabel("Fraud Probability")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
# Get column names from the training features
feature_columns = X.columns.tolist()

# Create a sample DataFrame with the correct column names and some dummy values
sample_data = {col: [0.0] for col in feature_columns}
sample = pd.DataFrame(sample_data)

pred = model.predict(sample)[0]
prob = model.predict_proba(sample)[0][1]

print("\n=== SAMPLE PREDICTION ===")
print("Fraud Prediction :", pred)
print("Fraud Probability:", prob)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle NaN values in y_train by dropping corresponding rows from both X_train and y_train
nan_mask_train = y_train.isna()
X_train = X_train[~nan_mask_train]
y_train = y_train[~nan_mask_train]

# Handle NaN values in y_test by dropping corresponding rows from both X_test and y_test
nan_mask_test = y_test.isna()
X_test = X_test[~nan_mask_test]
y_test = y_test[~nan_mask_test]

print(f"X_train shape before fit: {X_train.shape}")
print(f"y_train shape before fit: {y_train.shape}")

model = Pipeline(steps=[
    ('preprocessor', preprocess),
    ('classifier', RandomForestClassifier(random_state=42))
])

model.fit(X_train, y_train)
print("✔ Model trained successfully!")
