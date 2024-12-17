import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from statistics_helper import StatisticsHelper

# Load the data
full_employee_data = pd.read_csv('./full_employee_data_cleaned.csv')

# Prepare the data
target_column = 'Attrition'
X = full_employee_data.drop(target_column, axis=1)
y = full_employee_data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

stat_helper = StatisticsHelper(X, y, lr_model, y_test, y_pred)
stat_helper.show_regression_statistics()
