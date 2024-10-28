import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
data = pd.read_csv('data.csv', encoding='ISO-8859-1')  # Update filename if needed

# Encode categorical variables
categorical_columns = ['Country']  # Adjust based on dataset
for column in categorical_columns:
    data[column] = data[column].astype('category').cat.codes  # Encoding categorical variables

# Prepare feature and target variables
X = data[['UnitPrice', 'Quantity', 'Country']]  # Selected features
y = data['Quantity']  # Target variable - customer purchase quantity

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")

# Save the model
joblib.dump(model, 'recommendation_model.pkl')
print("Model saved as 'recommendation_model.pkl'")
