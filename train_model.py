import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_training_pipeline():
    print("Step 1: Loading 'car data.csv'...")
    try:
        df = pd.read_csv("car data.csv")
    except FileNotFoundError:
        print("CRITICAL ERROR: 'car data.csv' not found in the current directory.")
        return

    print("Step 2: Performing Feature Engineering...")
    # Engineer Age feature based on the current year (2026)
    df['Age'] = 2026 - df['Year']
    df.drop('Year', axis=1, inplace=True)
    
    # Drop Car_Name to protect model against high-cardinality overfitting
    df.drop('Car_Name', axis=1, inplace=True)

    # Transform categorical strings into mathematical flags using One-Hot Encoding
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Isolate Features and Target Variable
    X = df_encoded.drop('Selling_Price', axis=1)
    y = df_encoded['Selling_Price']

    # Stratify train-test split splits (80% training / 20% validation evaluation testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Step 3: Training Hyperparameter-Tuned Random Forest Regressor...")
    # Explicitly optimized parameters to prevent overfitting profiles
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    print("Step 4: Running Model Metrics Evaluation...")
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=============================================")
    print("         PRODUCTION MODEL REPORT             ")
    print("=============================================")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE) : {mse:.2f}")
    print(f"R-squared Score ($R^2$)   : {r2:.2f}")
    print("=============================================\n")

    # Generate feature importance distributions maps
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("Step 5: Exporting Analytics Visualizations...")
    # Plot A: Evaluation Prediction Concordance Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.7, color='#2980B9')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='#E74C3C', linewidth=2)
    plt.xlabel('Actual Prices (Lakhs)')
    plt.ylabel('Predicted Prices (Lakhs)')
    plt.title('Actual vs Predicted Model Behavior')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.close()

    # Plot B: Quantitative Feature Vector Weight Importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
    plt.title('Feature Weights Driving Car Appraisals')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    print("Step 6: Serializing Model Object Bundle File to Disk...")
    # Wrap model weights along with the feature schema structures to ensure data parity downstream
    model_bundle = {
        'model': rf_model,
        'features': list(X.columns)
    }
    with open('car_pricing_model.pkl', 'wb') as f:
        pickle.dump(model_bundle, f)
        
    print("SUCCESS: Pipeline execution complete. Output 'car_pricing_model.pkl' created.")

if __name__ == "__main__":
    run_training_pipeline()