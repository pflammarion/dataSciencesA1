import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats

#uploading dataset to python

data_live = pd.read_csv("Live_20210128.csv", delimiter=",")

#observations + variables

print(data_live)

print("Number of observations: ", data_live.shape[0])
print("\nNumber of variables: ", data_live.shape[1])

#removing the unecessary variables (does include id since it's the same as the index)

data_live.pop('status_id')
data_live.pop('Column1')
data_live.pop('Column2')
data_live.pop('Column3')
data_live.pop('Column4')

print("\nRemoved status_id, Column1, Column2, Column3, Column4")

print(data_live)

#calculating descriptive statistics for all the variables

print('\n Descriptive stats of the variable num_reactions:\n', data_live['num_reactions'].describe())
print('\n Descriptive stats of the variable num_comments:\n', data_live['num_comments'].describe())
print('\n Descriptive stats of the variable num_shares:\n',data_live['num_shares'].describe())
print('\n Descriptive stats of the variable num_likes:\n',data_live['num_likes'].describe())
print('\n Descriptive stats of the variable num_loves:\n',data_live['num_loves'].describe())
print('\n Descriptive stats of the variable num_wows:\n',data_live['num_wows'].describe())
print('\n Descriptive stats of the variable num_hahas:\n',data_live['num_hahas'].describe())
print('\n Descriptive stats of the variable num_sads:\n',data_live['num_sads'].describe())
print('\n Descriptive stats of the variable num_angrys:\n',data_live['num_angrys'].describe())




