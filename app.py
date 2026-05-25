import customtkinter as ctk
import pandas as pd
import pickle
from tkinter import messagebox

class DesktopAppraiserGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # System Window Geometry Properties
        self.title("AI Car Price Appraiser Engine")
        self.geometry("480x680")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Load Serialized Inference Bundle Check
        self.model, self.feature_columns = self.load_model_bundle()
        
        # Header Element Block Layout
        self.header_title = ctk.CTkLabel(self, text="Car Valuation Control Center", font=ctk.CTkFont(size=22, weight="bold"))
        self.header_title.pack(pady=(25, 5))
        
        if self.model is None:
            self.err_view = ctk.CTkLabel(self, text="Deployment Fail: 'car_pricing_model.pkl' missing.\nExecute 'train_model.py' first!", text_color="#E74C3C", font=ctk.CTkFont(size=14))
            self.err_view.pack(pady=40)
            return

        # Central Input Dynamic Interface Body Panel
        self.body_frame = ctk.CTkFrame(self)
        self.body_frame.pack(pady=15, padx=25, fill="both", expand=True)

        # Content Grid Rows
        self.setup_ui_fields()

    def load_model_bundle(self):
        try:
            with open('car_pricing_model.pkl', 'rb') as f:
                bundle = pickle.load(f)
                return bundle['model'], bundle['features']
        except FileNotFoundError:
            return None, None

    def setup_ui_fields(self):
        # Present Price Parameter
        ctk.CTkLabel(self.body_frame, text="Original Showroom Cost (In Lakhs):", anchor="w").pack(pady=(12, 0), padx=20, fill="x")
        self.ui_present_price = ctk.CTkEntry(self.body_frame, placeholder_text="e.g. 9.85")
        self.ui_present_price.pack(pady=(2, 10), padx=20, fill="x")

        # Manufacturing Timestamp parameter 
        ctk.CTkLabel(self.body_frame, text="Production/Model Year:", anchor="w").pack(pady=(10, 0), padx=20, fill="x")
        self.ui_year = ctk.CTkEntry(self.body_frame, placeholder_text="e.g. 2017")
        self.ui_year.pack(pady=(2, 10), padx=20, fill="x")

        # Total Structural Mileage usage profile
        ctk.CTkLabel(self.body_frame, text="Kilometers Logged (Odometer Reading):", anchor="w").pack(pady=(10, 0), padx=20, fill="x")
        self.ui_kms = ctk.CTkEntry(self.body_frame, placeholder_text="e.g. 25000")
        self.ui_kms.pack(pady=(2, 10), padx=20, fill="x")

        # Fuel Matrix selection profiles
        ctk.CTkLabel(self.body_frame, text="Engine Fuel Configuration Type:", anchor="w").pack(pady=(10, 0), padx=20, fill="x")
        self.val_fuel = ctk.StringVar(value="Petrol")
        self.ui_fuel = ctk.CTkOptionMenu(self.body_frame, values=["Petrol", "Diesel", "CNG"], variable=self.val_fuel)
        self.ui_fuel.pack(pady=(2, 10), padx=20, fill="x")

        # Transmission configurations mapping controls
        ctk.CTkLabel(self.body_frame, text="Gearbox Transmission Layout System:", anchor="w").pack(pady=(10, 0), padx=20, fill="x")
        self.val_trans = ctk.StringVar(value="Manual")
        self.ui_trans = ctk.CTkOptionMenu(self.body_frame, values=["Manual", "Automatic"], variable=self.val_trans)
        self.ui_trans.pack(pady=(2, 10), padx=20, fill="x")

        # Sale Origin Listing pathways
        ctk.CTkLabel(self.body_frame, text="Distribution Model Outlet Channel:", anchor="w").pack(pady=(10, 0), padx=20, fill="x")
        self.val_seller = ctk.StringVar(value="Dealer")
        self.ui_seller = ctk.CTkOptionMenu(self.body_frame, values=["Dealer", "Individual"], variable=self.val_seller)
        self.ui_seller.pack(pady=(2, 15), padx=20, fill="x")

        # Interactive Runtime Valuation Compute Action Target Button
        self.action_btn = ctk.CTkButton(self.body_frame, text="Run Predictive Intelligence Appraisal", command=self.compute_appraisal_inference, font=ctk.CTkFont(weight="bold"))
        self.action_btn.pack(pady=15, padx=20, fill="x")

        # Output Text Element Banner Display Block Layout object Frame
        self.ui_output_readout = ctk.CTkLabel(self.body_frame, text="", font=ctk.CTkFont(size=20, weight="bold"), text_color="#2ECC71")
        self.ui_output_readout.pack(pady=(5, 15))

    def compute_appraisal_inference(self):
        try:
            # Parse raw textual inputs into strict formats
            raw_showroom = float(self.ui_present_price.get())
            raw_year = int(self.ui_year.get())
            raw_kms = int(self.ui_kms.get())
            
            # Form structural check constraints handling edge boundary cases
            if raw_year > 2026 or raw_year < 1980:
                messagebox.showerror("Validation Boundary Alert", "Production data ranges out of scope. Enter a valid year between 1980 and 2026.")
                return
            if raw_showroom <= 0 or raw_kms < 0:
                messagebox.showerror("Validation Boundary Alert", "Physical quantitative variables (Price, Distance) cannot accept negative or zero entries.")
                return

            # Compute age relative to feature extraction reference year 2026
            computed_age = 2026 - raw_year
            
            # Construct dictionary mapping feature matrices
            feature_vector = {feat: 0 for feat in self.feature_columns}
            
            # Map values
            feature_vector['Present_Price'] = raw_showroom
            feature_vector['Driven_kms'] = raw_kms
            feature_vector['Age'] = computed_age
            feature_vector['Owner'] = 0  # Assuming standard baseline ownership tier model

            # Apply encoded properties mapping vectors 
            selected_fuel = self.val_fuel.get()
            if selected_fuel == "Diesel":
                feature_vector['Fuel_Type_Diesel'] = 1
            elif selected_fuel == "Petrol":
                feature_vector['Fuel_Type_Petrol'] = 1

            if self.val_trans.get() == "Manual":
                feature_vector['Transmission_Manual'] = 1

            if self.val_seller.get() == "Individual":
                feature_vector['Selling_type_Individual'] = 1

            # Convert map layout structure into standard internal prediction frame instance array row format
            evaluation_frame = pd.DataFrame([feature_vector])
            
            # Run model appraisal inference
            predicted_evaluation = self.model.predict(evaluation_frame)[0]
            
            # Enforce bounding floor limits physically to maintain reality checks
            if predicted_evaluation < 0:
                predicted_evaluation = 0.0

            # Output results rendering format to the UI system dashboard view port area
            self.ui_output_readout.configure(text=f"Estimated Residual Value: {predicted_evaluation:.2f} Lakhs")

        except ValueError:
            messagebox.showerror("Data Type Mismatch", "Input parameters parsing failure. Confirm that numeric metrics values are keyed correctly.")
        except Exception as err:
            messagebox.showerror("Runtime Processing Exception", f"Prediction execution failed: {str(err)}")

if __name__ == "__main__":
    app = DesktopAppraiserGUI()
    app.mainloop()