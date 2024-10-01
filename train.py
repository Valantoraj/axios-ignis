import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
#Read
data_train=pd.read_csv("D:\\axios\\train_features.csv")
data_test=pd.read_csv("D:\\axios\\test_features.csv")

#Independent-X Dependent-y
X_train=data_train.drop(columns=['sig_id','cp_type','cp_time','cp_dose'],axis='columns')
y_train=data_train['sig_id']
#Splitting data for testing and training
X_test=data_test.drop(columns=['sig_id','cp_type','cp_time','cp_dose'],axis='columns')
y_test=data_test['sig_id']
#Classifier
rf_classifier=RandomForestClassifier(n_estimators=10,random_state=42)
rf_classifier.fit(X_train,y_train)

#Predict
y_pred=rf_classifier.predict(X_test)

#Accuracy,Classification Report,Confusion
print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print('Classification Report of the model:')
print(classification_report(y_test,y_pred))
print('Confusion Matrix of the model:')
print(confusion_matrix(y_test,y_pred))

#Dump/Save model
joblib.dump(rf_classifier,'rf_classifier_model.pkl')