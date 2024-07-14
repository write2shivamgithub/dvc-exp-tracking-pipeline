import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import yaml

# Load train and test data
train = pd.read_csv('./data/raw/train.csv')
test = pd.read_csv('./data/raw/test.csv')

# Open the params
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

n_components = params['feature_engineering']['n_components']

# Separate feature and target
X_train = train.drop(columns=['Placed'])
y_train = train['Placed']
X_test = test.drop(columns=["Placed"])
y_test = test['Placed']

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Apply standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Combine processed feature and target
train_processed = pd.concat([pd.DataFrame(X_train_pca), pd.DataFrame(y_train).reset_index(drop=True)], axis=1)
test_processed = pd.concat([pd.DataFrame(X_test_pca), pd.DataFrame(y_test).reset_index(drop=True)], axis=1)

# Save the processed data
train_processed.to_csv('./data/processed/train_processed.csv', index=False)
test_processed.to_csv('./data/processed/test_processed.csv', index=False)
