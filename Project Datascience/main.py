import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats

#uploading dataset to python

df = pd.read_csv("Live_20210128.csv", delimiter=",")

#observations + variables

print(df)

print("Number of observations: ", df.shape[0])
print("\nNumber of variables: ", df.shape[1])

#removing the unecessary variables (does include id since it's the same as the index)

df.pop('Column1')
df.pop('Column2')
df.pop('Column3')
df.pop('Column4')

print("\nRemoved Column1, Column2, Column3, Column4")

print(df)

#calculating descriptive statistics for all the variables

print('\n Descriptive stats of the variable num_reactions:\n', df['num_reactions'].describe())
print('\n Descriptive stats of the variable num_comments:\n', df['num_comments'].describe())
print('\n Descriptive stats of the variable num_shares:\n',df['num_shares'].describe())
print('\n Descriptive stats of the variable num_likes:\n',df['num_likes'].describe())
print('\n Descriptive stats of the variable num_loves:\n',df['num_loves'].describe())
print('\n Descriptive stats of the variable num_wows:\n',df['num_wows'].describe())
print('\n Descriptive stats of the variable num_hahas:\n',df['num_hahas'].describe())
print('\n Descriptive stats of the variable num_sads:\n',df['num_sads'].describe())
print('\n Descriptive stats of the variable num_angrys:\n',df['num_angrys'].describe())

columns1 = ["num_reactions", "num_comments", "num_shares", "num_likes"]
columns2 = ["num_loves", "num_wows", "num_hahas", "num_sads", "num_angrys"]

df[columns1].boxplot()
plt.show()

df[columns2].boxplot()
plt.show()

counts_num_reactions = df["num_reactions"].value_counts()
counts_num_comments = df["num_comments"].value_counts()
counts_num_shares = df["num_shares"].value_counts()
counts_num_likes = df["num_likes"].value_counts()
counts_num_loves = df["num_loves"].value_counts()
counts_num_wows = df["num_wows"].value_counts()
counts_num_hahas = df["num_hahas"].value_counts()
counts_num_sads = df["num_sads"].value_counts()
counts_num_angrys = df["num_angrys"].value_counts()

counts_num_r = counts_num_reactions.index
counts_num_c = counts_num_comments.index
counts_num_s = counts_num_shares.index
counts_num_li = counts_num_likes.index
counts_num_lo = counts_num_loves.index
counts_num_w = counts_num_wows.index
counts_num_h = counts_num_hahas.index
counts_num_sa = counts_num_sads.index
counts_num_a = counts_num_angrys.index

values_num_r = counts_num_reactions.values
values_num_c = counts_num_comments.values
values_num_s = counts_num_shares.values
values_num_li = counts_num_likes.values
values_num_lo = counts_num_loves.values
values_num_w = counts_num_wows.values
values_num_h = counts_num_hahas.values
values_num_sa = counts_num_sads.values
values_num_a = counts_num_angrys.values


plt.scatter(counts_num_r, values_num_r, c='blue', label="reactions")
plt.show()
plt.scatter(counts_num_c, values_num_c, c='blue', label="comments")
plt.show()
plt.scatter(counts_num_s, values_num_s, c='blue', label="share")
plt.show()
plt.scatter(counts_num_li, values_num_li, c='blue', label="likes")
plt.show()
plt.scatter(counts_num_lo, values_num_lo, c='blue', label="loves")
plt.show()
plt.scatter(counts_num_w, values_num_w, c='blue', label="wows")
plt.show()
plt.scatter(counts_num_h, values_num_h, c='blue', label="hahas")
plt.show()
plt.scatter(counts_num_sa, values_num_sa, c='blue', label="sads")
plt.show()
plt.scatter(counts_num_a, values_num_a, c='blue', label="angrys")
plt.show()


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