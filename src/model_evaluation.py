import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dvclive import live
import os
import yaml
import json

# Load test data
test = pd.read_csv('./data/processed/test_processed.csv')

# Separate features and target
X_test = test.drop(columns = ['Placed'])
y_test = test['Placed']

# Load the trained model
rf = pickle.load(open('model.pkl','rb'))

# Make the prediction
y_pred = rf.predict(X_test)

# Calculation metrics
accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_pred,y_test)
f1 = f1_score(y_pred,y_test)

# Load parameter for logging
with open('params.yaml','r') as file :
    params = yaml.safe_load(file)

# Log metrics and parameters using dvclive
with live(save_dvs_exp=True) as live:
    live.log_metrics('accuracy', accuracy_score)
    live.log_metrics('precision', precision_score)
    live.log_metrics('recall', recall_score)
    live.log_metrics('f1', f1_score)

    for params, value in params.items():
        for key, val in value.items():
            live.log_params(f'{params}_{key}', val)

# Save metrics to a json file for compatibility with 
metrics = {
    'accuracy':accuracy,
    'precision':precision,
    'recall':recall,
    'f1_score':f1
}
with open("metrics.json",'w') as f:
    json.dump(metrics,f,indent=4)
    
