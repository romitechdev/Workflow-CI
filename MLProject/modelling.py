"""
modelling.py — MLProject (Workflow CI)
Dataset: Breast Cancer Wisconsin
"""

import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--random_state', type=int, default=42)
args = parser.parse_args()

BASE     = os.path.dirname(os.path.abspath(__file__))
train_df = pd.read_csv(os.path.join(BASE, 'cancer_preprocessing', 'cancer_train.csv'))
test_df  = pd.read_csv(os.path.join(BASE, 'cancer_preprocessing', 'cancer_test.csv'))

X_train = train_df.drop('target', axis=1)
y_train = train_df['target']
X_test  = test_df.drop('target', axis=1)
y_test  = test_df['target']

mlflow.sklearn.autolog()

model = RandomForestClassifier(
    n_estimators=args.n_estimators,
    random_state=args.random_state
)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
mlflow.log_metric("test_accuracy", acc)

print(f"Test Accuracy: {acc:.4f}")
print(classification_report(y_test, model.predict(X_test),
                             target_names=['malignant', 'benign']))
print("[DONE] Training selesai!")
