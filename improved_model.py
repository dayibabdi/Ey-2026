import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
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

# Base Features
BASE_FEATURES = [
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
    Robustly merges datasets with coordinate rounding and spectral ratios.
    """
    # 1. Standardize column names
    for df in [base_df, landsat_df, terra_df]:
        df.columns = df.columns.str.strip()
    
    # 2. COORDINATE ROUNDING: Critical for matching between different data sources.
    # Prevents silent join failures due to precision differences.
    for df in [base_df, landsat_df, terra_df]:
        df['Latitude'] = df['Latitude'].round(4)
        df['Longitude'] = df['Longitude'].round(4)
    
    # 3. Date Handling
    for df in [base_df, landsat_df, terra_df]:
        df['Sample Date'] = pd.to_datetime(df['Sample Date'], format='%d-%m-%Y', errors='coerce')

    # 4. Aggregation: Handle duplicates in satellite data
    land_agg = landsat_df.groupby(['Latitude', 'Longitude', 'Sample Date']).median().reset_index()
    terra_agg = terra_df.groupby(['Latitude', 'Longitude', 'Sample Date']).median().reset_index()

    # 5. Robust Merge (LEFT JOIN)
    print(f"Merging datasets... Base shape: {base_df.shape}")
    merged = pd.merge(base_df, land_agg, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
    merged = pd.merge(merged, terra_agg, on=['Latitude', 'Longitude', 'Sample Date'], how='left')
    print(f"Merged shape: {merged.shape}")
    
    # 6. Feature Engineering
    merged['Month'] = merged['Sample Date'].dt.month
    merged['DayOfYear'] = merged['Sample Date'].dt.dayofyear

    # 7. Spectral Ratios with Clipping (avoiding extreme outliers)
    with np.errstate(divide='ignore', invalid='ignore'):
        if 'nir' in merged.columns and 'green' in merged.columns:
            merged['Ratio_NIR_Green'] = (merged['nir'] / merged['green']).clip(0, 50)
        if 'nir' in merged.columns and 'swir16' in merged.columns:
            merged['Ratio_NIR_SWIR16'] = (merged['nir'] / merged['swir16']).clip(0, 50)
        if 'green' in merged.columns and 'swir22' in merged.columns:
            merged['Ratio_Green_SWIR22'] = (merged['green'] / merged['swir22']).clip(0, 50)
        if 'swir22' in merged.columns and 'swir16' in merged.columns:
            merged['Diff_SWIR'] = merged['swir22'] - merged['swir16']
        
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    
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

def get_base_model():
    """Returns a robust HistGradientBoostingRegressor Baseline."""
    return HistGradientBoostingRegressor(
        max_iter=1000, 
        learning_rate=0.05, 
        max_depth=10, 
        l2_regularization=0.1,
        random_state=42
    )

def train_and_evaluate(train_df):
    print("\n[STEP 3] Training & Evaluation (STABILIZED - NO LOG)...")
    
    train_df = train_df.sort_values(by="Sample Date").reset_index(drop=True)
    
    new_features = ["Ratio_NIR_Green", "Ratio_NIR_SWIR16", "Ratio_Green_SWIR22", "Diff_SWIR"]
    current_features = [f for f in (BASE_FEATURES + new_features) if f in train_df.columns]
    print(f"Features used: {current_features}")
    
    X_full = train_df[current_features]
    scores = {}

    for target in TARGETS:
        print(f"\n--- Modelling Target: {target} ---")
        y_full = train_df[target]

        # A) Time-Based Validation Split
        split_idx = int(len(train_df) * 0.8)
        X_train, X_val = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
        y_train, y_val = y_full.iloc[:split_idx], y_full.iloc[split_idx:]
        
        val_model = get_base_model()
        val_model.fit(X_train, y_train)
        
        y_pred = val_model.predict(X_val)
        y_pred = np.maximum(y_pred, 0) # Physical clipping
        
        r2 = r2_score(y_val, y_pred)
        print(f"Validation R2 Score: {r2:.4f}")
        scores[target] = r2

        # B) Final Training
        print(f"Retraining final model on full dataset...")
        final_model = get_base_model()
        final_model.fit(X_full, y_full)
        
        safe_name = target.replace(" ", "_").lower()
        model_path = os.path.join(MODEL_DIR, f"model_{safe_name}.joblib")
        joblib.dump(final_model, model_path)

    print("\nOverall Performance Summary (R2):")
    for t, s in scores.items():
        print(f"  {t}: {s:.4f}")
    print(f"  AVERAGE: {np.mean(list(scores.values())):.4f}")

def generate_submission(test_df):
    print("\n[STEP 4] Generating Submission...")
    
    new_features = ["Ratio_NIR_Green", "Ratio_NIR_SWIR16", "Ratio_Green_SWIR22", "Diff_SWIR"]
    current_features = [f for f in (BASE_FEATURES + new_features) if f in test_df.columns]
    X_test = test_df[current_features]
    
    submission_df = test_df[['Latitude', 'Longitude', 'Sample Date']].copy()
    
    # Ensure Date format is DD-MM-YYYY as in template
    submission_df['Sample Date'] = submission_df['Sample Date'].dt.strftime('%d-%m-%Y')
    
    for target in TARGETS:
        safe_name = target.replace(" ", "_").lower()
        model_path = os.path.join(MODEL_DIR, f"model_{safe_name}.joblib")
        
        if not os.path.exists(model_path): continue
            
        model = joblib.load(model_path)
        pred = model.predict(X_test)
        submission_df[target] = np.maximum(pred, 0)
    
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSUCCESS: Submission file generated at: {SUBMISSION_PATH}")

def main():
    setup_directories()
    train_df = prep_training_data()
    test_df = prep_prediction_data()
    if train_df is None or test_df is None: return
    train_and_evaluate(train_df)
    generate_submission(test_df)

if __name__ == "__main__":
    main()
