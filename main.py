import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Classification
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Encode non-numerical data
from sklearn.preprocessing import OneHotEncoder

# Stratified Sampling
from sklearn.model_selection import StratifiedShuffleSplit

# Load the data
general_data = pd.read_csv('datasets/general_data.csv')

##########################################
#                                        #
#   Data Exploration and Preprocessing   #
#                                        #
##########################################

# Display the first 5 rows of the data
print(general_data.head())

# Display the shape of the data
print(general_data.shape)

# Display the columns of the data
print(general_data.columns)

# Display the summary statistics of the data
print(general_data.describe(include='all'))

# Display the missing values in the data
print(general_data.isnull().sum())

# Display the unique values in the data
general_data.hist(bins=50, figsize=(20, 15))
plt.show()

###
# Remove useless columns
###

# EmployeeCount : All values are 1
# EmployeeID : Unique identifier
# Over18 : All values are 'Y'
# StandardHours : All values are 8
general_data = general_data.drop(['EmployeeCount', 'EmployeeID', 'Over18', 'StandardHours'], axis=1)


###
# Encode non-numerical data
###

cat_encoder = OneHotEncoder()
data_cat = general_data[['Department', 'Education', 'BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']]
data_cat_1hot = cat_encoder.fit_transform(data_cat)

# Display the encoded data
print(data_cat_1hot.toarray())

###
# Hierarchical Clustering
###

# Create stratified train-test split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# # Perform the split
# for train_index, test_index in split.split(general_data, general_data["income_cat"]):
#     # Use iloc instead of loc since split.split returns integer indices
#     strat_train_set = general_data.iloc[train_index]
#     strat_test_set = general_data.iloc[test_index]
#
# # Remove the income_cat column from both sets
# strat_train_set = strat_train_set.drop("income_cat", axis=1)
# strat_test_set = strat_test_set.drop("income_cat", axis=1)
#
# # Create a copy of the training set for further analysis
# general_data = strat_train_set.copy()