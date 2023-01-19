import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats

# uploading dataset to python

df = pd.read_csv("Live_20210128_clean.csv", delimiter=",")

# observations + variables

print(df)

print("Number of observations: ", df.shape[0])
print("\nNumber of variables: ", df.shape[1])

# calculating descriptive statistics for all the variables

print('\n Descriptive stats of the variable num_reactions:\n', df['num_reactions'].describe())
print('\n Descriptive stats of the variable num_comments:\n', df['num_comments'].describe())
print('\n Descriptive stats of the variable num_shares:\n', df['num_shares'].describe())
print('\n Descriptive stats of the variable num_likes:\n', df['num_likes'].describe())
print('\n Descriptive stats of the variable num_loves:\n', df['num_loves'].describe())
print('\n Descriptive stats of the variable num_wows:\n', df['num_wows'].describe())
print('\n Descriptive stats of the variable num_hahas:\n', df['num_hahas'].describe())
print('\n Descriptive stats of the variable num_sads:\n', df['num_sads'].describe())
print('\n Descriptive stats of the variable num_angrys:\n', df['num_angrys'].describe())

# 2.4 Linear Regression

## Perform an initial analysis of the variable num_shares based on the others
##by calculating the correlation coefficient between num_shares and each of the other variables ex-
##cept status_type, status_published and num_reactions. Which one is the most correlated with
##num_shares ?


corr_matrix = df.corr(numeric_only=True)

print('\nCorr matrix\n', corr_matrix)

correlation = corr_matrix['num_shares']

correlation = correlation.sort_values(ascending=False)

print('\nVariable which is correlated the most with the ozone content num_shares\n\n', correlation)


df_regsimple = smf.ols(formula='num_shares ~ num_loves', data=df).fit()
coef = df_regsimple.params
print('\nThe coefficients estimates are:\n', coef)


print('\nConfidence interval for the parameter β1\n', df_regsimple.conf_int(alpha=0.05))
