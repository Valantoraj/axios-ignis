import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import log_loss
import xgboost as xgb

# 1. Load Datasets
train_features = pd.read_csv('train_features.csv')
train_targets_scored = pd.read_csv('train_targets_scored.csv')
test_features = pd.read_csv('test_features.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# 2. Data Preprocessing

# Drop 'sig_id' column as it's just an identifier
train_features = train_features.drop('sig_id', axis=1)
test_features = test_features.drop('sig_id', axis=1)

# Separate features and target variables
X = train_features.drop(['cp_type', 'cp_time', 'cp_dose'], axis=1)
y = train_targets_scored.drop('sig_id', axis=1)

# One-hot encode categorical columns (cp_type, cp_time, cp_dose)
X = pd.get_dummies(train_features, columns=['cp_type', 'cp_time', 'cp_dose'])
X_test = pd.get_dummies(test_features, columns=['cp_type', 'cp_time', 'cp_dose'])

# Ensure that train and test sets have the same columns after encoding
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# Standardize the gene expression and cell viability data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training

# Initialize the XGBoost classifier
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Multi-output classifier to handle multiple MoA labels
multi_target_model = MultiOutputClassifier(xgb_model, n_jobs=-1)

# Train the model on the entire training dataset
multi_target_model.fit(X_scaled, y)

# 4. Predictions

# Predict probabilities on the test dataset
y_pred_proba = multi_target_model.predict_proba(X_test_scaled)

# For multi-label predictions, we need to concatenate the predictions for each MoA
# Since predict_proba returns a tuple of arrays, we must combine them
y_pred_combined = np.column_stack([y_pred_proba[i][:, 1] for i in range(len(y_pred_proba))])

# 5. Prepare Submission

# Load the sample submission to keep the same format
submission = pd.DataFrame(y_pred_combined, columns=train_targets_scored.columns[1:])
submission.insert(0, 'sig_id', sample_submission['sig_id'])
submission.to_csv('submission.csv', index=False)

print("Submission saved!")