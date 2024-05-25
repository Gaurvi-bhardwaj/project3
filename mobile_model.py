import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
data = pd.read_csv('mobile_data.csv')

# Feature columns
features = ['battery', 'bluetooth', 'processor_speed', 'dual_sim', 'front_camera', '4g', 'internal_memory', 'depth', 'weight', 'cores']

# Target column
target = 'price'

# Split the data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'mobile_price_model.pkl')
