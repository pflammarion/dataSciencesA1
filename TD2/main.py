import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
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

#ex3

import statsmodels.api as sm
lm = sm.OLS.from_formula('maxO3 ~ Ne12', oz)
oz_regsimple = lm. fit ()
print (oz_regsimple.summary())

plt.scatter(oz.Ne12, oz.maxO3)

slope = np.polyfit(oz.Ne12, oz.maxO3, 1)[0]
intercept = np.polyfit(oz.Ne12, oz.maxO3, 1)[1]

#plt.plot([x,x], [x,x], 'k−', color = 'r') #plot 2 points of the fitted line
def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

abline(slope, intercept)
