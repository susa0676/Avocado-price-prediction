import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv("avocado.csv")

# Encode the 'region' column
region_encoder = LabelEncoder()
data['region'] = region_encoder.fit_transform(data['region'])

# Convert 'Date' to datetime and extract additional features
data['Date'] = pd.to_datetime(data['Date'])
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day_of_week'] = data['Date'].dt.dayofweek

# Prepare features and target
X = data[['year', 'month', 'day_of_week', 'Total Volume', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'region']]
y = data['AveragePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Initialize the RandomForestRegressor model
rf_model = RandomForestRegressor(random_state=42)

# Perform randomized search with cross-validation
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, 
                                   n_iter=100, cv=3, scoring='neg_mean_absolute_error', 
                                   verbose=1, random_state=42, n_jobs=-1)

random_search.fit(X_train, y_train)

# Display the best parameters and train the model with them
print("Best parameters found: ", random_search.best_params_)
best_rf_model = random_search.best_estimator_

# Evaluate the model
y_pred = best_rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Save the best model and the encoder
joblib.dump(best_rf_model, 'best_random_forest_model.pkl')
joblib.dump(region_encoder, 'region_encoder.pkl')

print("Model and encoder saved as 'best_random_forest_model.pkl' and 'region_encoder.pkl'")
