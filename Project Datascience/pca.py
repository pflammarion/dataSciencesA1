import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Live_20210128_clean.csv", delimiter=",")
#select only 4 col
data = df[["num_comments", "num_shares", "num_likes", "num_loves"]]
variance = data.var()
print("\n Variance: \n", variance)
variance.plot(kind='bar')
plt.xlabel("Columns")
plt.ylabel("Variance")
plt.title("Variance of columns")
plt.show()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled_df = pd.DataFrame(data_scaled, index=data.index, columns=data.columns)

#putting mean to 0 and std to 1#

print(data_scaled_df.describe())
pca = PCA()
pca.fit(data_scaled)

loadings = pca.components_.T
print("\nLoadings vectors associated to each principal component: \n", loadings)


explained_variance = np.round(pca.explained_variance_ratio_*100, decimals=1)
cumulative_explained_variance = [explained_variance[:n + 1].sum() for n in range(len(explained_variance))]

print('\nExplained Variance: ', explained_variance)
print('\nCumulative explaned variance: ', cumulative_explained_variance)

plt.plot(explained_variance, label='PVE by each component')
plt.plot(cumulative_explained_variance, label='Cumulative PVE')
plt.legend()
plt.ylabel("Percentage of variance explained (PVE)")
plt.xlabel("Principal component")
plt.title("Scree plot 4 components")
plt.show()

plt.bar(range(len(explained_variance)), explained_variance)
plt.title("Explained Variance")
plt.ylabel("Percentage of variance explained (PVE)")
plt.xlabel("Principal component")
plt.show()


pca = PCA(n_components=2)
pca.fit(data_scaled)

loadings = pca.components_.T
print("\nLoadings vectors associated to the two first principal component:  \n", loadings)

explained_variance = np.round(pca.explained_variance_ratio_*100, decimals=1)
cumulative_explained_variance = [explained_variance[:n + 1].sum() for n in range(len(explained_variance))]

print('\nExplained Variance: ', explained_variance)
print('\nCumulative explaned variance: ', cumulative_explained_variance)



fig, axis = plt.subplots(figsize=(7, 7))
axis.set_xlim(-1, 1)
axis.set_ylim(-1, 1)
plt.plot([-1, 1], [0, 0], color="silver", linestyle="-", linewidth=1)
plt.plot([0, 0], [-1, 1], color="silver", linestyle="-", linewidth=1)
for j in range(0, 4):
    plt.arrow(0, 0, loadings[j, 0], loadings[j, 1],
              head_width=0.02, width=0.001, color="red")
    plt.annotate(data_scaled_df.columns[j], (loadings[j, 0], loadings[j, 1]))
cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
axis.add_artist(cercle)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()


scores = pca.transform(data_scaled)

plt.scatter(scores[:, 0], scores[:, 1], s=5)
plt.title("Live, PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

fig, axis = plt.subplots(figsize=(7, 7))
axis.set_xlim(-1, 1)
axis.set_ylim(-1, 1)
plt.plot([-1, 1], [0, 0], color="silver", linestyle="-", linewidth=1)
plt.plot([0, 0], [-1, 1], color="silver", linestyle="-", linewidth=1)
for j in range(0, 4):
    plt.arrow(0, 0, loadings[j, 0], loadings[j, 1],
              head_width=0.02, width=0.001, color="red")
    plt.annotate(data_scaled_df.columns[j], (loadings[j, 0], loadings[j, 1]))
cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
axis.add_artist(cercle)
plt.scatter(scores[:, 0], scores[:, 1], s=5)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()
