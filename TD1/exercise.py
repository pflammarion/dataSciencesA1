import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


# Load the dataset
data = pd.read_table("exercice.csv", delimiter=";")

# Select the columns you want to standardize
columns = ['X1', 'X2']

# Standardize the selected columns
scaler = StandardScaler()
scaler.fit(data[columns])
data[columns] = scaler.transform(data[columns])

# Calculate the variance matrix
variance_matrix = data[columns].cov()

#Diagonalize the variance matrix
eigenvalues, eigenvectors = np.linalg.eig(variance_matrix)

# Find the principal component loading vectors
loading_vectors = eigenvectors * eigenvalues

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


data = pd.read_table("exercice.csv", delimiter=";")


##for i, col in enumerate(columns):
    #data[col] = data[col].apply(lambda x: x * loading_vectors[i])

#print(data['X1'])




