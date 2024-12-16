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
employee_survey_data = pd.read_csv('datasets/employee_survey_data.csv')
manager_survey_data = pd.read_csv('datasets/manager_survey_data.csv')

employee_data = None


##########################################
#                                        #
#   Data Exploration and Preprocessing   #
#                                        #
##########################################

def merge_employee_data():
    global employee_data
    employee_data = general_data.merge(employee_survey_data, on='EmployeeID')
    employee_data = employee_data.merge(manager_survey_data, on='EmployeeID')


merge_employee_data()


def create_working_time_columns():
    """
        Process in_time.csv and out_time.csv data to create working time columns in the general_data dataframe
    """
    global general_data
    in_time = pd.read_csv('datasets/in_time.csv').astype('datetime64[ns]')
    out_time = pd.read_csv('datasets/out_time.csv').astype('datetime64[ns]')

    print(in_time.head())

    average = (out_time - in_time)

    # Convert to hours
    average = average.loc[:, :] / np.timedelta64(1, 'h')

    print(average.head())

    working_time_df = pd.DataFrame()

    # Create a column EmployeeID
    working_time_df['EmployeeID'] = in_time.iloc[:, 0]
    working_time_df['EmployeeID'] = working_time_df['EmployeeID'].astype('int64')

    # Create a column min and max
    working_time_df['MinimumWorkingTime'] = average.mask(average <= 0).min(axis=1)
    working_time_df['MaximumWorkingTime'] = average.max(axis=1)

    # Create a column average
    working_time_df['AverageWorkingTime'] = average.mean(axis=1)

    # Merge the working time data with the general data
    general_data = general_data.merge(working_time_df, on='EmployeeID')


create_working_time_columns()

######################
#  Data exploration  #
######################

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

######################
# Data preprocessing #
######################

###
# Remove useless columns
###

# EmployeeCount : All values are 1
# Over18 : All values are 'Y'
# StandardHours : All values are 8
general_data = general_data.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1)

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
