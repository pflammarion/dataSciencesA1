from itertools import combinations
from statistics import LinearRegression

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

print('Summary report of the fitting simple\n', df_regsimple.summary())

# 2.3 Principal Component Analysis (PCA)


# Select only the desired columns
data = df[["num_comments", "num_shares", "num_likes", "num_loves"]]

print('Variance of each variables:\n', data.var())

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Create the PCA object
pca = PCA()

# Fit the PCA model to the data
pca.fit(data_scaled)
print('PCA', pca)

# Get the principal component loading vectors
#loading_vectors = pca.components_

#print("\nLoading vectors: \n", loading_vectors)


# Get the explained variance ratio of each principal component
explained_variance = pca.explained_variance_ratio_
print('\nPVE: \n', explained_variance)
# Get the cumulative explained variance ratio
cumulative_explained_variance = [explained_variance[:n + 1].sum() for n in range(len(explained_variance))]

# Plot the explained variance and cumulative explained variance
plt.plot(explained_variance, label='PVE by each component')
plt.plot(cumulative_explained_variance, label='Cumulative PVE')
plt.legend()
plt.show()

pca = PCA(n_components=2)

# Fit the PCA model to the data
pca.fit(data_scaled)
print('PCA', pca)

# Get the principal component loading vectors
loading_vectors = pca.components_

# Get the principal component scores
scores = pca.transform(data_scaled)

# Create the biplot
plt.scatter(scores[:, 0], scores[:, 1], s=5)

# Add the loading vectors to the plot
plt.quiver(np.zeros(2), np.zeros(2), loading_vectors[:, 0], loading_vectors[:, 1], angles='xy', scale_units='xy',
           scale=1)

# Add the correlation circle
circle = plt.Circle((0, 0), 1, color='gray', fill=False)
plt.gca().add_artist(circle)

# Add labels and show the plot
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
print('Per_Var: ',per_var)
print('cumsum',np.cumsum(per_var))
labels = ['PC1', 'PC2']
plt.plot([1, 2], per_var)
plt.ylabel("percentage of variance explained (PVE)")
plt.xlabel("Principal component")
plt.title("Scree plot")


loadings = pca.components_.T
print('loading\n', loadings)
fig, axis = plt.subplots(figsize=(5, 5))
axis.set_xlim(-1, 1)
axis.set_ylim(-1, 1)
plt.plot([-1, 1], [0, 0], color="silver", linestyle="-", linewidth=1)
plt.plot([0, 0], [-1, 1], color="silver", linestyle="-", linewidth=1)
for j in range(0, 4):
    plt.arrow(0, 0, loadings[j, 0], loadings[j, 1],
              head_width=0.02, width=0.001, color="red")
    plt.annotate(df.columns[j], (loadings[j, 0], loadings[j, 1]))
cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
axis.add_artist(cercle)
plt.show()