# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data
# Assuming `full_employee_data` is already loaded as a DataFrame
full_employee_data: pd.DataFrame = pd.read_csv('full_employee_data_cleaned.csv')

# Prepare the data
target_column = 'Attrition_Yes'
# Replace 'target_column' with the actual name of the target column
X = full_employee_data.drop([target_column, 'Attrition_No'], axis=1)
y = full_employee_data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))