import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# URL to dataset
url = "https://raw.githubusercontent.com/write2shivamgithub/StudentData/master/student_data.csv"

# Read the dataset
df = pd.read_csv(url)

# Convert 'No' and 'Yes' to 0 and 1 in y_pred (if necessary)
df = df.replace({'No': 0, 'Yes': 1})

# Split the data into training & testing sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

#Save the splits
train.to_csv('./data/raw/train.csv', index=False)
test.to_csv('./data/raw/test.csv', index=False)
