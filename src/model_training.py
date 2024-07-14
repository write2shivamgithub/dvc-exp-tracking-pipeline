import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import yaml
import os

# Load parameters
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

n_estimators = params['model_training']['n_estimators']
max_depth = params['model_training']['max_depth']
bootstrap = params['model_training']['bootstrap']
criterion = params['model_training']['criterion']

# Load processed data
train = pd.read_csv('./data/processed/train_processed.csv')

# Separate feature and target
X_train = train.drop(columns=['Placed'])
y_train = train['Placed']

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Train the RandomForest model
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, criterion=criterion)
rf.fit(X_train_imputed, y_train)

# Save the model
pickle.dump(rf, open('model.pkl', 'wb'))
