# %% [markdown]

# # PCA
# 
# *This is a markdown cell. All lines need to be commented with a #, else the linter goes mental.*
# 
# There must be something that makes this nicer in VS Code....google that later. Or maybe I should just use a jupyter notebook instead...
#
# This is going to be my working space for PCA.


# %%

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Data for this exercise
from sklearn.datasets import load_breast_cancer

#%%
# Load data
breast_data = load_breast_cancer().data
breast_labels = load_breast_cancer().target
features = load_breast_cancer().feature_names
# %%
# Adjust data
labels = np.reshape(breast_labels,(569,1))
final_breast_data = np.concatenate([breast_data,labels],axis=1)
dataset = pd.DataFrame(final_breast_data)
# %%
# Name columns
features_labels = np.append(features,'label')
dataset.columns = features_labels

# Apply label description text
dataset['label'].replace(0, 'Benign',inplace=True)
dataset['label'].replace(1, 'Malignant',inplace=True)

# %% [markdown]
## Now the actual PCA!

#%%
# Scale the data
X = dataset.loc[:, features].values

scaler = StandardScaler()
scaler.fit(X)

scaled_X = scaler.transform(X)

# %%
pca = PCA(n_components=2)
pca.fit(scaled_X)
X_new = pca.transform(scaled_X)
# %%
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
print('Total explained variationt: {}'.format(np.sum(pca.explained_variance_ratio_)))

# %%
# How many components?
pca_full = PCA().fit(scaled_X)

plt.figure(figsize=(10,7))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 30)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axvline(1, c='b')
plt.axhline(0.9, c='r')
plt.grid()
plt.show()

# %%
