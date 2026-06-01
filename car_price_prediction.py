import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import customtkinter as ctk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================================
# PART 1: ANALYTICS & MODEL TRAINING ENGINE
# =====================================================================
def train_and_save_model():
    print("\n--- INITIATING AI TRAINING PIPELINE ---")
    if not os.path.exists("car data.csv"):
        print("CRITICAL ERROR: 'car data.csv' not found in the current directory.")
        return False

    print("Step 1: Loading and Engineering Data...")
    df = pd.read_csv("car data.csv")
    
    # Engineer Age and drop unnecessary columns
    df['Age'] = 2026 - df['Year']
    df.drop(['Year', 'Car_Name'], axis=1, inplace=True)
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop('Selling_Price', axis=1)
    y = df_encoded['Selling_Price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Step 2: Training Hyperparameter-Tuned Model...")
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    print("Step 3: Running Model Metrics Evaluation...")
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=============================================")
    print("         PRODUCTION MODEL REPORT             ")
    print("=============================================")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE) : {mse:.2f}")
    print(f"R-squared Score (R²)     : {r2:.2f}")
    print("=============================================\n")

    print("Step 4: Exporting Analytics Visualizations...")
    
    # Chart A: Actual vs Predicted
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

    # Chart B: Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
    plt.title('Feature Weights Driving Car Appraisals')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    print("Step 5: Serializing Model to Disk...")
    model_bundle = {
        'model': rf_model,
        'features': list(X.columns)
    }
    with open('car_pricing_model.pkl', 'wb') as f:
        pickle.dump(model_bundle, f)
        
    print("SUCCESS: Training complete. 'car_pricing_model.pkl' created.\n")
    return True


# =====================================================================
# PART 2: DESKTOP APPLICATION GUI
# =====================================================================
class DesktopAppraiserGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # System Window Geometry Properties
        self.title("AI Car Price Appraiser Engine")
        self.geometry("480x680")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Load Model (Will train automatically if missing)
        self.model, self.feature_columns = self.load_model_bundle()
        
        # Header Element
        self.header_title = ctk.CTkLabel(self, text="Car Valuation Control Center", font=ctk.CTkFont(size=22, weight="bold"))
        self.header_title.pack(pady=(25, 5))
        
        # Error handling if BOTH model and csv are missing
        if self.model is None:
            self.err_view = ctk.CTkLabel(self, text="Deployment Fail: Missing model and 'car data.csv'.\nPlease place the dataset in this folder.", text_color="#E74C3C", font=ctk.CTkFont(size=14))
            self.err_view.pack(pady=40)
            return

        # Central Input Panel
        self.body_frame = ctk.CTkFrame(self)
        self.body_frame.pack(pady=15, padx=25, fill="both", expand=True)

        self.setup_ui_fields()

    def load_model_bundle(self):
        # Auto-train logic if model is missing
        if not os.path.exists('car_pricing_model.pkl'):
            print("Model not found on startup. Initializing training module...")
            success = train_and_save_model()
            if not success:
                return None, None

        # Load the serialized model
        try:
            with open('car_pricing_model.pkl', 'rb') as f:
                bundle = pickle.load(f)
                return bundle['model'], bundle['features']
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None

    def setup_ui_fields(self):
        # Present Price
        ctk.CTkLabel(self.body_frame, text="Original Showroom Cost (In Lakhs):", anchor="w").pack(pady=(12, 0), padx=20, fill="x")
        self.ui_present_price = ctk.CTkEntry(self.body_frame, placeholder_text="e.g. 9.85")
        self.ui_present_price.pack(pady=(2, 10), padx=20, fill="x")

        # Manufacturing Year
        ctk.CTkLabel(self.body_frame, text="Production/Model Year:", anchor="w").pack(pady=(10, 0), padx=20, fill="x")
        self.ui_year = ctk.CTkEntry(self.body_frame, placeholder_text="e.g. 2017")
        self.ui_year.pack(pady=(2, 10), padx=20, fill="x")

        # Kilometers
        ctk.CTkLabel(self.body_frame, text="Kilometers Logged (Odometer Reading):", anchor="w").pack(pady=(10, 0), padx=20, fill="x")
        self.ui_kms = ctk.CTkEntry(self.body_frame, placeholder_text="e.g. 25000")
        self.ui_kms.pack(pady=(2, 10), padx=20, fill="x")

        # Fuel Type
        ctk.CTkLabel(self.body_frame, text="Engine Fuel Configuration Type:", anchor="w").pack(pady=(10, 0), padx=20, fill="x")
        self.val_fuel = ctk.StringVar(value="Petrol")
        self.ui_fuel = ctk.CTkOptionMenu(self.body_frame, values=["Petrol", "Diesel", "CNG"], variable=self.val_fuel)
        self.ui_fuel.pack(pady=(2, 10), padx=20, fill="x")

        # Transmission
        ctk.CTkLabel(self.body_frame, text="Gearbox Transmission Layout System:", anchor="w").pack(pady=(10, 0), padx=20, fill="x")
        self.val_trans = ctk.StringVar(value="Manual")
        self.ui_trans = ctk.CTkOptionMenu(self.body_frame, values=["Manual", "Automatic"], variable=self.val_trans)
        self.ui_trans.pack(pady=(2, 10), padx=20, fill="x")

        # Seller Type
        ctk.CTkLabel(self.body_frame, text="Distribution Model Outlet Channel:", anchor="w").pack(pady=(10, 0), padx=20, fill="x")
        self.val_seller = ctk.StringVar(value="Dealer")
        self.ui_seller = ctk.CTkOptionMenu(self.body_frame, values=["Dealer", "Individual"], variable=self.val_seller)
        self.ui_seller.pack(pady=(2, 15), padx=20, fill="x")

        # Action Button
        self.action_btn = ctk.CTkButton(self.body_frame, text="Run Predictive Intelligence Appraisal", command=self.compute_appraisal_inference, font=ctk.CTkFont(weight="bold"))
        self.action_btn.pack(pady=15, padx=20, fill="x")

        # Output Readout
        self.ui_output_readout = ctk.CTkLabel(self.body_frame, text="", font=ctk.CTkFont(size=20, weight="bold"), text_color="#2ECC71")
        self.ui_output_readout.pack(pady=(5, 15))

    def compute_appraisal_inference(self):
        try:
            raw_showroom = float(self.ui_present_price.get())
            raw_year = int(self.ui_year.get())
            raw_kms = int(self.ui_kms.get())
            
            # Constraints
            if raw_year > 2026 or raw_year < 1980:
                messagebox.showerror("Validation Alert", "Enter a valid year between 1980 and 2026.")
                return
            if raw_showroom <= 0 or raw_kms < 0:
                messagebox.showerror("Validation Alert", "Price and Distance must be positive numbers.")
                return

            computed_age = 2026 - raw_year
            
            # Map Feature Vectors
            feature_vector = {feat: 0 for feat in self.feature_columns}
            
            feature_vector['Present_Price'] = raw_showroom
            feature_vector['Driven_kms'] = raw_kms
            feature_vector['Age'] = computed_age
            feature_vector['Owner'] = 0  
            
            if self.val_fuel.get() == "Diesel" and 'Fuel_Type_Diesel' in feature_vector:
                feature_vector['Fuel_Type_Diesel'] = 1
            elif self.val_fuel.get() == "Petrol" and 'Fuel_Type_Petrol' in feature_vector:
                feature_vector['Fuel_Type_Petrol'] = 1

            if self.val_trans.get() == "Manual" and 'Transmission_Manual' in feature_vector:
                feature_vector['Transmission_Manual'] = 1

            if self.val_seller.get() == "Individual" and 'Selling_type_Individual' in feature_vector:
                feature_vector['Selling_type_Individual'] = 1

            # Run Inference
            evaluation_frame = pd.DataFrame([feature_vector])
            predicted_evaluation = self.model.predict(evaluation_frame)[0]
            
            if predicted_evaluation < 0:
                predicted_evaluation = 0.0

            self.ui_output_readout.configure(text=f"Estimated Residual Value: ₹ {predicted_evaluation:.2f} Lakhs")

        except ValueError:
            messagebox.showerror("Data Type Mismatch", "Please ensure numbers are keyed correctly.")
        except Exception as err:
            messagebox.showerror("Runtime Exception", f"Execution failed: {str(err)}")

# =====================================================================
# PART 3: EXECUTION ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    app = DesktopAppraiserGUI()
    app.mainloop()
