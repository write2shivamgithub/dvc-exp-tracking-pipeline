import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import yaml

# Load train and test data
train = pd.read_csv('./data/raw/train.csv')
test = pd.read_csv('./data/raw/test.csv')

# Opening the params
with open('params.yaml','r') as file:
    params = yaml.safe_load(file)

n_components = params['feature_engineering']['n_components']

# Separate feature and target
X_train = train.drop(columns = ['Placed'])
y_train = test['Placed']
X_test = test.drop(columns = ["Placed"])
y_test = test['Placed']

# Apply standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Combine processed feature and target
train_processed = pd.concat([pd.DataFrame(X_train_pca, y_train)], axis=1)
test_processed = pd.concat([pd.DataFrame(X_test_pca, y_test)], axis=1)

# Save the processed data
train_processed.to_csv('./data/processed/train_processed.csv', index=False)
test_processed.to_csv('./data/processed/test_processed.csv', index=False)
