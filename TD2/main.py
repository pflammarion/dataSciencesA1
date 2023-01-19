import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats
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

#import statsmodels.api as sm
#lm = sm.OLS.from_formula('maxO3 ~ Ne12', oz)
#oz_regsimple = lm. fit ()
#print (oz_regsimple.summary())

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

#ex4

oz_regsimple = smf.ols(formula='maxO3 ~ Ne12', data=oz).fit()
print('\nConfidence interval for the parameter β1\n', oz_regsimple.conf_int(alpha=0.1))

#ex5

oz_regmult_all = smf.ols(formula='maxO3 ~ Ne12 + maxO3v + T12', data=oz).fit()
print('\nSummary report of the fitting\n', oz_regmult_all.summary())

#ex6

oz_regmult = smf.ols(formula='maxO3 ~ Ne12 + maxO3v', data=oz).fit()
print('\nSummary report of the fitting\n', oz_regmult.summary())

#H0: The slope of the regression line is equal to zero.
#To perform the zero slope hypothesis test, you can use the f_regression function
# from the sklearn.feature_selection module.
# This function returns the F-value and p-value for each predictor in the model.
# The null hypothesis for this test is that the slope of the predictor is equal to zero,
# and the alternative hypothesis is that the slope is not equal to zero.

# calculate the t-value and p-value for the predictor Ne12
t_value, p_value = stats.ttest_1samp(oz.Ne12, 0)

# set the significance level (alpha)
alpha = 0.05

# compare the p-value to the significance level
if p_value < alpha:
    print("Reject the null hypothesis. There is a relationship between Ne12 and maxO3.")
else:
    print("Fail to reject the null hypothesis. There is no relationship between Ne12 and maxO3.")


