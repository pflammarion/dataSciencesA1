import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import statsmodels.formula.api as smf

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

#ex4

oz_regsimple = smf.ols(formula='maxO3 ~ Ne12', data=oz).fit()
print('\nConfidence interval for the parameter β1\n', oz_regsimple.conf_int(alpha=0.1))

#ex5

oz_regmult = smf.ols(formula='maxO3 ~ Ne12', data=oz).fit()
print('\nSummary report of the fitting\n', oz_regmult.summary())
## n'est pas la bonne commande => reg simple




