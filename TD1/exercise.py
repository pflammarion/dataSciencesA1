import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


# Load the dataset
data = pd.read_table("exercice.csv", delimiter=";")

# Select the columns you want to standardize
columns_to_standardize = ['X1', 'X2']

# Standardize the selected columns
scaler = StandardScaler()
scaler.fit(data[columns_to_standardize])
data[columns_to_standardize] = scaler.transform(data[columns_to_standardize])

# Calculate the variance matrix
variance_matrix = data[columns_to_standardize].cov()

#Diagonalize the variance matrix
eigenvalues, eigenvectors = np.linalg.eig(variance_matrix)

# Find the principal component loading vectors
loading_vectors = eigenvectors / eigenvalues

print('\nStandardized\n')
print(data)
print('\nVariance\n')
print(variance_matrix)

print('\neigenvalues\n')
print(eigenvalues)

print('\neigenvectors\n')
print(eigenvectors)

print('\nloading_vectors\n')
print(loading_vectors)




