import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import log_loss
import xgboost as xgb
train_features = pd.read_csv('train_features.csv')
train_targets_scored = pd.read_csv('train_targets_scored.csv')
test_features = pd.read_csv('test_features.csv')
sample_submission = pd.read_csv('sample_submission.csv'
train_features = train_features.drop('sig_id', axis=1)
test_features = test_features.drop('sig_id', axis=1)
X = train_features.drop(['cp_type', 'cp_time', 'cp_dose'], axis=1)
y = train_targets_scored.drop('sig_id', axis=1)
X = pd.get_dummies(train_features, columns=['cp_type', 'cp_time', 'cp_dose'])
X_test = pd.get_dummies(test_features, columns=['cp_type', 'cp_time', 'cp_dose'])
X_test = X_test.reindex(columns=X.columns, fill_value=0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
multi_target_model = MultiOutputClassifier(xgb_model, n_jobs=-1)
multi_target_model.fit(X_scaled, y)
y_pred_proba = multi_target_model.predict_proba(X_test_scaled)
y_pred_combined = np.column_stack([y_pred_proba[i][:, 1] for i in range(len(y_pred_proba))])
submission = pd.DataFrame(y_pred_combined, columns=train_targets_scored.columns[1:])
submission.insert(0, 'sig_id', sample_submission['sig_id'])
submission.to_csv('submission.csv', index=False)

print("Submission saved!")