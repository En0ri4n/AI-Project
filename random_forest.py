# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from statistics_helper import StatisticsHelper

# Load the data
# Assuming `full_employee_data` is already loaded as a DataFrame
full_employee_data: pd.DataFrame = pd.read_csv('full_employee_data_cleaned.csv')

# Prepare the data
target_column = 'Attrition'
# Replace 'target_column' with the actual name of the target column
X = full_employee_data.drop(target_column, axis=1)
y = full_employee_data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

stats_helper = StatisticsHelper(X, y, rf_model, y_test, y_pred)
stats_helper.show_accuracy()
stats_helper.show_classification_report()
stats_helper.show_confusion_matrix()
stats_helper.show_cross_val_score()
stats_helper.show_roc_auc_score()
