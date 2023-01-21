from itertools import combinations
from statistics import LinearRegression

import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("Live_20210128_clean.csv", delimiter=",")
#filter dataframe
df.drop('status_published', inplace=True, axis=1)
df.drop('num_reactions', inplace=True, axis=1)


corr_matrix = df.corr(numeric_only=True)

correlation = corr_matrix['num_shares']

correlation = correlation.sort_values(ascending=False)

plt.figure(figsize=(13, 6))
sns.heatmap(corr_matrix, vmax=1, annot=True, linewidths=.5)
plt.xticks(rotation=30, horizontalalignment='right')
plt.title("Correlation matrix")
plt.show()

print('\nVariable which is correlated the most with the ozone content num_shares\n\n', correlation)

df_regsimple = smf.ols(formula='num_shares ~ num_loves', data=df).fit()

#1
coef = df_regsimple.params
print('\nThe coefficients estimates are:\n', coef)

#2
conf_int = df_regsimple.conf_int(alpha=0.05)
print('\nConfidence interval for the parameter β1\n', conf_int)

#3

se = df_regsimple.bse['num_loves']

t_value = coef / se

# Calculate the p-value
p_value = (1 - stats.t.cdf(abs(t_value), df_regsimple.df_resid)) * 2

# check p-value

#4
print('\n summary() of simple linear regression: \n', df_regsimple.summary())

df1 = df[df['status_type'] == 'video']
df2 = df[df['status_type'] == 'photo']

# extract intercept b and slope m
plt.scatter(df1['num_loves'], df1['num_shares'], label='Video')
plt.scatter(df2['num_loves'], df2['num_shares'], color='purple', label='Photo')


b, m = df_regsimple.params

# plot y = m*x + b

plt.axline(xy1=(0, b), linestyle='--', color='red', slope=m, label=f'$y = {m:.1f}x {b:+.1f}$')
#plt.axline(xy1=(0, b), color='blue', slope=conf_int[0]['num_loves'])
#plt.axline(xy1=(0, b), color='purple', slope=conf_int[1]['num_loves'])
plt.legend()
plt.title('Regression simple')
plt.xlabel('Num_loves')
plt.ylabel('Num_shares')
plt.show()

