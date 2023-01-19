import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
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

df_regsimple = smf.ols(formula='num_shares ~ num_loves', data=df).fit()
coef = df_regsimple.params

conf_int = df_regsimple.conf_int(alpha=0.05)

print(conf_int)
if (conf_int.loc['num_loves'][0] < 0) & (conf_int.loc['num_loves'][1] > 0):
    print("The coefficient is not significantly non-zero, there is no significant impact of the predictor on the number of shares")
else:
    print("The coefficient is significantly non-zero, there is an impact of the predictor on the number of shares")

print(df_regsimple.summary())


#Multiple linear regression
df.pop('status_type')
df.pop('status_published')
df.pop('status_id')
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
plt.bar(r2_dict.keys(), [sum(r2_dict[i])/len(r2_dict[i]) for i in r2_dict.keys()])
plt.xlabel('Number of Features')
plt.ylabel('Average R^2')
plt.show()

# print the best model
print("The best model has", best_num_features, "features and an R^2 value of", max(r2_scores))
print("The features in the best model are:", best_feature_list)

