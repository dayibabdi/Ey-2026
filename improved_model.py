import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import joblib

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = r"c:\Users\hp\Desktop\Ey challenge\Jupyter Notebook Package\Jupyter Notebook Package"
PROCESSED_DIR = r"c:\Users\hp\Desktop\Ey challenge\processed_data"
MODEL_DIR = r"c:\Users\hp\Desktop\Ey challenge\models"
SUBMISSION_PATH = r"c:\Users\hp\Desktop\Ey challenge\submission.csv"

# Input Files
WQA_TRAIN = os.path.join(DATA_DIR, "water_quality_training_dataset.csv")
LANDSAT_TRAIN = os.path.join(DATA_DIR, "landsat_features_training.csv")
TERRA_TRAIN = os.path.join(DATA_DIR, "terraclimate_features_training.csv")

SUB_TEMP = os.path.join(DATA_DIR, "submission_template.csv")
LANDSAT_VAL = os.path.join(DATA_DIR, "landsat_features_validation.csv")
TERRA_VAL = os.path.join(DATA_DIR, "terraclimate_features_validation.csv")

# Targets
TARGETS = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]

# Features (Optimization: Added Lat/Lon, Dates, Spectral Indices)
FEATURES = [
    "Latitude", "Longitude", 
    "nir", "green", "swir16", "swir22", 
    "NDMI", "MNDWI", "pet", 
    "Month", "DayOfYear"
]

def setup_directories():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess_and_merge(base_df, landsat_df, terra_df, is_submission=False):
    """
    Robustly merges datasets and extracts date features.
    NO median imputation - lets HistGradientBoostingRegressor handle NaNs natively.
    """
    # 1. Standardize column names
    for df in [base_df, landsat_df, terra_df]:
        df.columns = df.columns.str.strip()
    
    # 2. Convert 'Sample Date' to datetime
    for df in [base_df, landsat_df, terra_df]:
        df['Sample Date'] = pd.to_datetime(df['Sample Date'], format='%d-%m-%Y', errors='coerce')

    # 3. Robust Merge (Inner Join)
    # We join base_df with landsat, then with terra
    print(f"Merging datasets... Initial base shape: {base_df.shape}")
    merged = pd.merge(base_df, landsat_df, on=['Latitude', 'Longitude', 'Sample Date'], how='inner')
    merged = pd.merge(merged, terra_df, on=['Latitude', 'Longitude', 'Sample Date'], how='inner')
    print(f"Merged shape: {merged.shape}")
    
    # 4. Feature Engineering
    merged['Month'] = merged['Sample Date'].dt.month
    merged['DayOfYear'] = merged['Sample Date'].dt.dayofyear
    
    # Optimization: Do NOT fillna. HistGradientBoostingRegressor handles it better.
    
    return merged

def prep_training_data():
    print("\n[STEP 1] Preparing Training Data...")
    try:
        df_base = pd.read_csv(WQA_TRAIN)
        df_land = pd.read_csv(LANDSAT_TRAIN)
        df_terra = pd.read_csv(TERRA_TRAIN)
    except FileNotFoundError as e:
        print(f"Error loading training files: {e}")
        return None

    train_final = preprocess_and_merge(df_base, df_land, df_terra)
    
    output_path = os.path.join(PROCESSED_DIR, "train_final.csv")
    train_final.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return train_final

def prep_prediction_data():
    print("\n[STEP 2] Preparing Prediction Data (Validation/Test)...")
    try:
        df_sub = pd.read_csv(SUB_TEMP)
        df_land = pd.read_csv(LANDSAT_VAL)
        df_terra = pd.read_csv(TERRA_VAL)
    except FileNotFoundError as e:
        print(f"Error loading validation files: {e}")
        return None

    test_final = preprocess_and_merge(df_sub, df_land, df_terra, is_submission=True)
    
    output_path = os.path.join(PROCESSED_DIR, "test_final.csv")
    test_final.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return test_final

def train_and_evaluate(train_df):
    print("\n[STEP 3] Training & Evaluation...")
    
    # Sort by date for proper time-based splitting
    train_df = train_df.sort_values(by="Sample Date").reset_index(drop=True)
    
    # Verify features exist
    available_features = [f for f in FEATURES if f in train_df.columns]
    print(f"Features used: {available_features}")
    
    X_full = train_df[available_features]
    
    # Storage for detailed scores
    scores = {}

    for target in TARGETS:
        print(f"\n--- Modelling Target: {target} ---")
        y_full = train_df[target]

        # ---------------------------------------------------------
        # A) Time-Based Validation Split (Last 20% by time)
        # ---------------------------------------------------------
        split_idx = int(len(train_df) * 0.8)
        X_train, X_val = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
        y_train, y_val = y_full.iloc[:split_idx], y_full.iloc[split_idx:]
        
        # HistGradientBoostingRegressor natively handles NaNs (missing values)
        # and doesn't require scaling.
        val_model = HistGradientBoostingRegressor(
            max_iter=1000, 
            learning_rate=0.05, 
            max_depth=10, 
            random_state=42
        )
        val_model.fit(X_train, y_train)
        
        y_pred = val_model.predict(X_val)
        
        # Metrics
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        print(f"Validation Results (Time-Based Split):")
        print(f"  R2 Score: {r2:.4f} (Target: 1.0)")
        print(f"  RMSE:     {rmse:.4f}")
        print(f"  MAE:      {mae:.4f}")
        
        scores[target] = r2

        # ---------------------------------------------------------
        # B) Final Training on ALL Data
        # ---------------------------------------------------------
        print(f"Retraining on full dataset ({len(train_df)} rows)...")
        final_model = HistGradientBoostingRegressor(
            max_iter=1500,  # Increased slightly for final model
            learning_rate=0.04, 
            max_depth=12, 
            random_state=42
        )
        final_model.fit(X_full, y_full)
        
        # Save Model
        safe_name = target.replace(" ", "_").lower()
        model_path = os.path.join(MODEL_DIR, f"model_{safe_name}.joblib")
        joblib.dump(final_model, model_path)
        print(f"Model saved to: {model_path}")

    print("\nOverall Performance Summary (R2):")
    for t, s in scores.items():
        print(f"  {t}: {s:.4f}")
    print(f"  AVERAGE: {np.mean(list(scores.values())):.4f}")

def generate_submission(test_df):
    print("\n[STEP 4] Generating Submission...")
    
    if test_df is None or test_df.empty:
        print("Error: No prediction data available.")
        return

    # Prepare features
    available_features = [f for f in FEATURES if f in test_df.columns]
    X_test = test_df[available_features]
    
    # We need to maintain the original rows for the submission file
    submission_df = test_df[['Latitude', 'Longitude', 'Sample Date']].copy()
    
    # Load models and predict
    for target in TARGETS:
        safe_name = target.replace(" ", "_").lower()
        model_path = os.path.join(MODEL_DIR, f"model_{safe_name}.joblib")
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found for {target} at {model_path}")
            continue
            
        model = joblib.load(model_path)
        submission_df[target] = model.predict(X_test)
    
    # Format Sample Date back to string if needed for submission format
    # The original template uses 'DD-MM-YYYY' usually, let's ensure we match the input format roughly
    # submission_df['Sample Date'] = submission_df['Sample Date'].dt.strftime('%d-%m-%Y') 
    
    # Save
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSUCCESS: Submission file generated at: {SUBMISSION_PATH}")

def main():
    setup_directories()
    
    # 1. Prep Data
    train_df = prep_training_data()
    test_df = prep_prediction_data()
    
    if train_df is None or test_df is None:
        print("Stopping due to data loading errors.")
        return
        
    # 2. Train & Eval
    train_and_evaluate(train_df)
    
    # 3. Predict
    generate_submission(test_df)

if __name__ == "__main__":
    main()
