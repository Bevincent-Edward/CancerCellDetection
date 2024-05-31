# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load your dataset
can = pd.read_csv("cell_samples.csv")

# Preprocess the dataset
can.replace('?', np.nan, inplace=True)
can = can.apply(pd.to_numeric, errors='ignore')
can.fillna(can.mean(), inplace=True)
can.drop_duplicates(inplace=True)

# Define features and target variable
x = can.drop('Class', axis=1)
y = can['Class']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Train the model
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

# Save the model
joblib.dump(knn, 'CancerCellDetection.pkl')
