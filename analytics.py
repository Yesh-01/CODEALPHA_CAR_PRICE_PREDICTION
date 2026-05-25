import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load the dataset
print("Loading 'car data.csv'...")
df = pd.read_csv("car data.csv")

# 2. Feature Engineering
# Calculate car age based on the current year (2026)
df['Age'] = 2026 - df['Year']
df.drop('Year', axis=1, inplace=True)

# Drop 'Car_Name' because it has too many unique values for a small dataset
df.drop('Car_Name', axis=1, inplace=True)

# Convert categorical text columns into numbers using One-Hot Encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# 3. Separate Features (X) and Target Price (y)
X = df_encoded.drop('Selling_Price', axis=1)
y = df_encoded['Selling_Price']

# Split data: 80% for training the AI, 20% for testing it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Regression Model
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- MODEL PERFORMANCE METRICS ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} (Average prediction deviation)")
print(f"Mean Squared Error (MSE) : {mse:.2f}")
print(f"R-squared Score ($R^2$)   : {r2:.2f} (Closer to 1.0 means higher accuracy)")

# Extract feature importance metrics
feature_importance = pd.DataFrame({
    'Feature': X.columns, 
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- FEATURE IMPORTANCE RANKS ---")
print(feature_importance.to_string(index=False))

# 6. Generate and Save Charts
print("\nGenerating evaluation charts...")

# Chart A: Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', linewidth=2)
plt.xlabel('Actual Selling Price (Lakhs)')
plt.ylabel('Predicted Selling Price (Lakhs)')
plt.title('Actual vs Predicted Car Prices')
plt.grid(True)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()

# Chart B: Horizontal Bar Chart of Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Car Price Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("Done! Charts saved as 'actual_vs_predicted.png' and 'feature_importance.png'.")