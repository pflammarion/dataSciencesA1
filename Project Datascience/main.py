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
