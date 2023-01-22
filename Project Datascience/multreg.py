import inline as inline
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib
import matplotlib.pyplot as plt
import warnings
import numpy as np
import statsmodels.formula.api as smf
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import scipy.stats as stats
import statsmodels.api as sm
from sklearn import linear_model
import plotly.express as px
# For data visualization
import seaborn as sns

#uploading dataset to python

df = pd.read_csv("Live_20210128.csv", delimiter=",")

#observations + variables

print(df)

#removing the unecessary variables (does include id since it's the same as the index)

df.pop('Column1')
df.pop('Column2')
df.pop('Column3')
df.pop('Column4')
df.pop('status_id')
df.pop('status_type')
df.pop('status_published')

print("\nRemoved Column1, Column2, Column3, Column4")

#Multiple linear regression
# create the X and y arrays

X = df.drop(['num_shares'], axis=1)
y = df['num_shares']

# create lists to store the R^2 values and feature lists for each number of features
r2_scores = []
feature_lists = []

# perform subset selection for each possible number of features
for i in range(1, 7):
    comb = combinations(X.columns, i)
    for feature_list in comb:
        feature_list = "+".join(feature_list)
        formula = 'num_shares ~' + feature_list
        model = smf.ols(formula=formula, data=df).fit()
        r2_scores.append(model.rsquared_adj)
        feature_lists.append(feature_list)

# find the index of the best model (the one with the highest R^2 value)
best_index = r2_scores.index(max(r2_scores))
best_num_features = len(feature_lists[best_index])
best_feature_list = feature_lists[best_index]

# plot the R^2 values versus the number of features

num_features = [len(fl.split("+")) for fl in feature_lists]

r2_dict = {}
for i, r2 in zip(num_features, r2_scores):
    if i not in r2_dict:
        r2_dict[i] = []
    r2_dict[i].append(r2)

# plot the R^2 values as a bar plot
plt.plot(r2_dict.keys(), [sum(r2_dict[i])/len(r2_dict[i]) for i in r2_dict.keys()], c='red')
plt.bar(r2_dict.keys(), [sum(r2_dict[i])/len(r2_dict[i]) for i in r2_dict.keys()])
plt.xlabel('Number of Features')
plt.ylabel('Average R^2')
plt.show()

# print the best model
print("The best model has", best_num_features, "features and an R^2 value of", max(r2_scores))
print("The features in the best model are:", best_feature_list)


model = smf.ols(formula='num_shares ~ num_loves + num_wows', data=df).fit()
df['num_shares_pred'] = model.predict(df[['num_loves', 'num_wows']])
fig = px.scatter_3d(df, x='num_loves', y='num_wows', z='num_shares_pred',
              color='num_shares_pred',size='num_shares_pred',title="Multiple Linear Regression")
fig.show()
model = smf.ols(formula='num_shares ~ num_loves + num_likes', data=df).fit()
df['num_shares_pred'] = model.predict(df[['num_loves', 'num_likes']])
fig = px.scatter_3d(df, x='num_loves', y='num_likes', z='num_shares_pred',
              color='num_shares_pred',size='num_shares_pred',title="Multiple Linear Regression")
fig.show()
model = smf.ols(formula='num_shares ~ num_loves + num_comments', data=df).fit()
df['num_shares_pred'] = model.predict(df[['num_loves', 'num_comments']])
fig = px.scatter_3d(df, x='num_loves', y='num_comments', z='num_shares_pred',
              color='num_shares_pred',size='num_shares_pred',title="Multiple Linear Regression")
fig.show()
model = smf.ols(formula='num_shares ~ num_loves + num_hahas', data=df).fit()
df['num_shares_pred'] = model.predict(df[['num_loves', 'num_hahas']])
fig = px.scatter_3d(df, x='num_loves', y='num_hahas', z='num_shares_pred',
              color='num_shares_pred',size='num_shares_pred',title="Multiple Linear Regression")
fig.show()
model = smf.ols(formula='num_shares ~ num_loves + num_sads', data=df).fit()
df['num_shares_pred'] = model.predict(df[['num_loves', 'num_sads']])
fig = px.scatter_3d(df, x='num_loves', y='num_sads', z='num_shares_pred',
              color='num_shares_pred',size='num_shares_pred',title="Multiple Linear Regression")
fig.show()

model = smf.ols(formula='num_shares ~ num_loves + num_wows + num_hahas + num_sads + num_comments', data=df).fit()
# obtain the coefficient estimates
coef = model.params
print(coef)
r_squared = model.rsquared
print(r_squared)

# obtain the t-statistics and p-values for each coefficient
tvalues = model.tvalues[1:]
pvalues = model.pvalues[1:]

# set the significance level
alpha = 0.05

# compare the p-values to the significance level
for i, p in enumerate(pvalues):
    if p < alpha:
        print(f'Coefficient {model.params.index[i+1]} is statistically significant')
    else:
        print(f'Coefficient {model.params.index[i+1]} is not statistically significant')
