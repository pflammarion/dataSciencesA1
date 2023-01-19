import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load the dataset
oz = pd.read_csv("ozone.csv", delimiter=" ")

#ex1

print("Number of observations: ", oz.shape[0])
print("\nNumber of variables: ", oz.shape[1])

#ex2

print("\nDescriptive statistics\n", oz.describe())

scatter_matrix(oz, alpha=0.2, figsize=(6, 6), diagonal='kde', color="red")
plt.show()

corr_matrix = oz.corr()

print('\nCorr matrix\n', corr_matrix)

correlation = corr_matrix['maxO3']

correlation = correlation.sort_values(ascending=False)

print('\nVariable which is correlated the most with the ozone content maxO3\n\n', correlation)



