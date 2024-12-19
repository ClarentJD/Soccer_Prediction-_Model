# -*- coding: utf-8 -*-
"""Soccer_Prediction"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas
# %pip install numpy
# %pip install pandas scikit-learn
# %pip install pandas xgboost

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel


# Load the dataset
df = pd.read_csv('/content/drive/MyDrive/E0.csv')

# Drop rows with NaN values in any of the feature columns or target columns
df.dropna(subset=['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HST', 'AST', 'HC', 'AC', 'HS', 'AS', 'FTR'], inplace=True)

# Encode FTR as categorical data
result_mapping = {'D': 1, 'A': 0, 'H': 2}
df['FTR'] = df['FTR'].map(result_mapping)

# Select features and target variables
features = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HST', 'AST', 'HC', 'AC', 'HS', 'AS']
X = df[features]
y_FTR = df['FTR']
y_FTHG = df['FTHG']
y_FTAG = df['FTAG']
y_HTHG = df['HTHG']
y_HTAG = df['HTAG']
y_HC = df['HC']
y_AC = df['AC']

# Keep only the necessary columns for the final output
df = df[['HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HST', 'AST', 'HC', 'AC', 'HS', 'AS']]

# Feature importance to select top features
feature_selector = SelectFromModel(XGBClassifier(eval_metric='mlogloss'))
feature_selector.fit(X, y_FTR)

# Get selected features
selected_features = X.columns[feature_selector.get_support()]
print("Selected features:", selected_features)
X = X[selected_features]

# Split the data into training and testing sets
X_train, X_test, y_FTR_train, y_FTR_test = train_test_split(X, y_FTR, test_size=0.2, random_state=42)
_, _, y_FTHG_train, y_FTHG_test = train_test_split(X, y_FTHG, test_size=0.2, random_state=42)
_, _, y_FTAG_train, y_FTAG_test = train_test_split(X, y_FTAG, test_size=0.2, random_state=42)
_, _, y_HTHG_train, y_HTHG_test = train_test_split(X, y_HTHG, test_size=0.2, random_state=42)
_, _, y_HTAG_train, y_HTAG_test = train_test_split(X, y_HTAG, test_size=0.2, random_state=42)
_, _, y_HC_train, y_HC_test = train_test_split(X, y_HC, test_size=0.2, random_state=42)
_, _, y_AC_train, y_AC_test = train_test_split(X, y_AC, test_size=0.2, random_state=42)

# Initialize the XGBoost Classifier and Regressors
xgb_classifier = XGBClassifier(eval_metric='mlogloss')
xgb_regressor = XGBRegressor()

# Parameters for Grid Search
param_grid_classifier = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}
param_grid_regressor = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}

# Grid Search for Classifier
grid_search_classifier = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid_classifier, cv=5, scoring='accuracy')
grid_search_classifier.fit(X, y_FTR)
best_classifier = grid_search_classifier.best_estimator_

# Grid Search for Regressor
grid_search_regressor_FTHG = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid_regressor, cv=5, scoring='neg_mean_squared_error')
grid_search_regressor_FTAG = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid_regressor, cv=5, scoring='neg_mean_squared_error')
grid_search_regressor_HTHG = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid_regressor, cv=5, scoring='neg_mean_squared_error')
grid_search_regressor_HTAG = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid_regressor, cv=5, scoring='neg_mean_squared_error')
grid_search_regressor_HC = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid_regressor, cv=5, scoring='neg_mean_squared_error')
grid_search_regressor_AC = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid_regressor, cv=5, scoring='neg_mean_squared_error')

# Train the regressors
grid_search_regressor_FTHG.fit(X, y_FTHG)
grid_search_regressor_FTAG.fit(X, y_FTAG)
grid_search_regressor_HTHG.fit(X, y_HTHG)
grid_search_regressor_HTAG.fit(X, y_HTAG)
grid_search_regressor_HC.fit(X, y_HC)
grid_search_regressor_AC.fit(X, y_AC)

# Best models after Grid Search
best_regressor_FTHG = grid_search_regressor_FTHG.best_estimator_
best_regressor_FTAG = grid_search_regressor_FTAG.best_estimator_
best_regressor_HTHG = grid_search_regressor_HTHG.best_estimator_
best_regressor_HTAG = grid_search_regressor_HTAG.best_estimator_
best_regressor_HC = grid_search_regressor_HC.best_estimator_
best_regressor_AC = grid_search_regressor_AC.best_estimator_

# Cross-validate the classifier and regressors
cv_classifier = cross_val_score(best_classifier, X, y_FTR, cv=5, scoring='accuracy')
cv_regressor_FTHG = cross_val_score(best_regressor_FTHG, X, y_FTHG, cv=5, scoring='neg_mean_squared_error')
cv_regressor_FTAG = cross_val_score(best_regressor_FTAG, X, y_FTAG, cv=5, scoring='neg_mean_squared_error')
cv_regressor_HTHG = cross_val_score(best_regressor_HTHG, X, y_HTHG, cv=5, scoring='neg_mean_squared_error')
cv_regressor_HTAG = cross_val_score(best_regressor_HTAG, X, y_HTAG, cv=5, scoring='neg_mean_squared_error')
cv_regressor_HC = cross_val_score(best_regressor_HC, X, y_HC, cv=5, scoring='neg_mean_squared_error')
cv_regressor_AC = cross_val_score(best_regressor_AC, X, y_AC, cv=5, scoring='neg_mean_squared_error')

print(f'Cross-Validation Accuracy (FTR): {cv_classifier.mean():.2f}')
print(f'Cross-Validation MSE (FTHG): {-cv_regressor_FTHG.mean():.2f}')
print(f'Cross-Validation MSE (FTAG): {-cv_regressor_FTAG.mean():.2f}')
print(f'Cross-Validation MSE (HTHG): {-cv_regressor_HTHG.mean():.2f}')
print(f'Cross-Validation MSE (HTAG): {-cv_regressor_HTAG.mean():.2f}')
print(f'Cross-Validation MSE (HC): {-cv_regressor_HC.mean():.2f}')
print(f'Cross-Validation MSE (AC): {-cv_regressor_AC.mean():.2f}')

# Train the best models
best_classifier.fit(X_train, y_FTR_train)
best_regressor_FTHG.fit(X_train, y_FTHG_train)
best_regressor_FTAG.fit(X_train, y_FTAG_train)
best_regressor_HTHG.fit(X_train, y_HTHG_train)
best_regressor_HTAG.fit(X_train, y_HTAG_train)
best_regressor_HC.fit(X_train, y_HC_train)
best_regressor_AC.fit(X_train, y_AC_train)

# Make predictions on the test set
y_FTR_pred = best_classifier.predict(X_test)
y_FTHG_pred = best_regressor_FTHG.predict(X_test)
y_FTAG_pred = best_regressor_FTAG.predict(X_test)
y_HTHG_pred = best_regressor_HTHG.predict(X_test)
y_HTAG_pred = best_regressor_HTAG.predict(X_test)
y_HC_pred = best_regressor_HC.predict(X_test)
y_AC_pred = best_regressor_AC.predict(X_test)

# Calculate the accuracy and error
accuracy_FTR = accuracy_score(y_FTR_test, y_FTR_pred)
mse_FTHG = mean_squared_error(y_FTHG_test, y_FTHG_pred)
mse_FTAG = mean_squared_error(y_FTAG_test, y_FTAG_pred)
mse_HTHG = mean_squared_error(y_HTHG_test, y_HTHG_pred)
mse_HTAG = mean_squared_error(y_HTAG_test, y_HTAG_pred)
mse_HC = mean_squared_error(y_HC_test, y_HC_pred)
mse_AC = mean_squared_error(y_AC_test, y_AC_pred)

print(f'Accuracy (FTR): {accuracy_FTR:.2f}')
print(f'MSE (FTHG): {mse_FTHG:.2f}')
print(f'MSE (FTAG): {mse_FTAG:.2f}')
print(f'MSE (HTHG): {mse_HTHG:.2f}')
print(f'MSE (HTAG): {mse_HTAG:.2f}')
print(f'MSE (HC): {mse_HC:.2f}')
print(f'MSE (AC): {mse_AC:.2f}')

# Ensure the feature set for prediction matches the training set
last_three_matches = df.tail(3)[selected_features]

# Predict the results for the last three matches
predictions_FTR = best_classifier.predict(last_three_matches)
predictions_FTHG = best_regressor_FTHG.predict(last_three_matches)
predictions_FTAG = best_regressor_FTAG.predict(last_three_matches)
predictions_HTHG = best_regressor_HTHG.predict(last_three_matches)
predictions_HTAG = best_regressor_HTAG.predict(last_three_matches)
predictions_HC = best_regressor_HC.predict(last_three_matches)
predictions_AC = best_regressor_AC.predict(last_three_matches)

# Map numerical predictions back to results
inverse_result_mapping = {1: 'D', 0: 'A', 2: 'H'}
predicted_results_FTR = [inverse_result_mapping[pred] for pred in predictions_FTR]

# Add predictions to the final output
final_output = df.tail(3).copy()
final_output['Predicted FTR'] = predicted_results_FTR
final_output['Predicted FTHG'] = predictions_FTHG
final_output['Predicted FTAG'] = predictions_FTAG
final_output['Predicted HTHG'] = predictions_HTHG
final_output['Predicted HTAG'] = predictions_HTAG
final_output['Predicted HC'] = predictions_HC
final_output['Predicted AC'] = predictions_AC

# Map numerical results back to text for full-time result
full_time_result_mapping = {1: 'D', 0: 'A', 2: 'H'}
final_output['FTR'] = final_output['FTR'].map(full_time_result_mapping)

# Display the final output
print("Predicted results for the last three matches:")
print(final_output[['HomeTeam', 'AwayTeam', 'FTR', 'Predicted FTR', 'Predicted FTHG', 'Predicted FTAG', 'Predicted HTHG', 'Predicted HTAG', 'Predicted HC', 'Predicted AC']])

# Save the final output to a CSV file
final_output.to_csv('predicted_results.csv', index=False)



