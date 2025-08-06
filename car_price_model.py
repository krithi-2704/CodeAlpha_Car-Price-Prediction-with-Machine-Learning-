import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ----- Ensure path works every time -----
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "car data.csv")
data = pd.read_csv(file_path)

print(data.head())

car_df = data.copy()

# --- Data Cleaning ---
car_df.rename(columns={
    "Selling_Price": "market_value",
    "Driven_kms": "odometer_reading"
}, inplace=True)

car_df["age_in_years"] = 2025 - car_df["Year"]
car_df["brand_name"] = car_df["Car_Name"].str.split().str[0].str.lower()
car_df.drop(["Car_Name", "Year"], axis=1, inplace=True)

car_df["gearbox_type"] = car_df["Transmission"].map({"Manual": 0, "Automatic": 1})
car_df.drop("Transmission", axis=1, inplace=True)

car_df = pd.get_dummies(car_df, columns=["Fuel_Type", "Selling_type", "brand_name"], drop_first=True)

print(car_df.head())

# --- Model Prep ---
input_features = car_df.drop("market_value", axis=1)
target_price = car_df["market_value"]

X_train, X_test, y_train, y_test = train_test_split(
    input_features, target_price, test_size=0.2, random_state=42
)

# --- Linear Regression ---
price_model = LinearRegression()
price_model.fit(X_train, y_train)
predictions = price_model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"R² Score      : {r2:.3f}")
print(f"MAE (Error)   : {mae:.2f}")
print(f"RMSE (Error)  : {rmse:.2f}")

# --- Random Forest ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

r2_rf = r2_score(y_test, rf_predictions)
mae_rf = mean_absolute_error(y_test, rf_predictions)
rmse_rf = np.sqrt(mean_squared_error(y_test, rf_predictions))

print("\n📊 Random Forest Results:")
print(f"R² Score      : {r2_rf:.3f}")
print(f"MAE (Error)   : {mae_rf:.2f}")
print(f"RMSE (Error)  : {rmse_rf:.2f}")

# --- Gradient Boosting ---
boosted_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
boosted_model.fit(X_train, y_train)
gbr_predictions = boosted_model.predict(X_test)

r2_gbr = r2_score(y_test, gbr_predictions)
mae_gbr = mean_absolute_error(y_test, gbr_predictions)
rmse_gbr = np.sqrt(mean_squared_error(y_test, gbr_predictions))

print("\n⚡ Gradient Boosting Results:")
print(f"R² Score      : {r2_gbr:.3f}")
print(f"MAE (Error)   : {mae_gbr:.2f}")
print(f"RMSE (Error)  : {rmse_gbr:.2f}")

# --- Save Actual vs Predicted Plot ---
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=rf_predictions)
plt.xlabel("Actual Price (in Lakhs)")
plt.ylabel("Predicted Price (in Lakhs)")
plt.title("Actual vs Predicted Car Prices - Random Forest")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(current_dir, "actual_vs_predicted_rf.png"))
plt.close()

# --- Save Feature Importance Plot ---
importances = rf_model.feature_importances_
feature_names = input_features.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title("Top 10 Important Features in Price Prediction")
plt.xlabel("Relative Importance")
plt.ylabel("Feature")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(current_dir, "top_features_rf.png"))
plt.close()

# --- Save Error Distribution Plot ---
errors = y_test - rf_predictions

plt.figure(figsize=(8,4))
sns.histplot(errors, bins=20, kde=True)
plt.title("Prediction Error Distribution - Random Forest")
plt.xlabel("Error (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(current_dir, "rf_error_distribution.png"))
plt.close()
