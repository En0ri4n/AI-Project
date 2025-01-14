import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from statistics_helper import StatisticsHelper

# Load the data
full_employee_data: pd.DataFrame = pd.read_csv('./full_employee_data_cleaned.csv')

target_column = 'Attrition'

# Prepare the data
X = full_employee_data.drop(target_column, axis=1)
y = full_employee_data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Display the statistics
stats_helper = StatisticsHelper(X_train, y_train, rf_model, y_test, y_pred)
stats_helper.show_all()
