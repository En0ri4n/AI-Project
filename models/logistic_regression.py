import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from statistics_helper import StatisticsHelper

# Load the data
full_employee_data = pd.read_csv('./full_employee_data_cleaned.csv')

# Prepare the data
target_column = 'Attrition'
X = full_employee_data.drop(target_column, axis=1)
y = full_employee_data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Initialize the Logistic Regression model
log_reg_model = LogisticRegression(random_state=42, max_iter=1000)  # Increase max_iter if convergence warnings occur
log_reg_model.fit(X_train, y_train)

# Make predictions
y_pred = log_reg_model.predict(X_test)

# Display the statistics
stat_helper = StatisticsHelper(X, y, log_reg_model, y_test, y_pred)
stat_helper.show_all()
