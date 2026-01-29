import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Set paths
DATA_DIR = r"c:\Users\hp\Desktop\Ey challenge\Jupyter Notebook Package\Jupyter Notebook Package"
MODEL_DIR = r"c:\Users\hp\Desktop\Ey challenge\models"
os.makedirs(MODEL_DIR, exist_ok=True)

WQA_TRAIN = os.path.join(DATA_DIR, "water_quality_training_dataset.csv")
LANDSAT_TRAIN = os.path.join(DATA_DIR, "landsat_features_training.csv")
TERRA_TRAIN = os.path.join(DATA_DIR, "terraclimate_features_training.csv")

SUB_TEMP = os.path.join(DATA_DIR, "submission_template.csv")
LANDSAT_VAL = os.path.join(DATA_DIR, "landsat_features_validation.csv")
TERRA_VAL = os.path.join(DATA_DIR, "terraclimate_features_validation.csv")

def preprocess_and_merge(base_df, landsat_df, terra_df, is_submission=False):
    """
    Robustly merges datasets on ['Latitude', 'Longitude', 'Sample Date'].
    Extracts date features.
    """
    # 1. Standardize column names (strip whitespace just in case)
    for df in [base_df, landsat_df, terra_df]:
        df.columns = df.columns.str.strip()
    
    # 2. Convert 'Sample Date' to datetime in ALL dataframes
    # This is critical for the merge to work correctly
    for df in [base_df, landsat_df, terra_df]:
        df['Sample Date'] = pd.to_datetime(df['Sample Date'], format='%d-%m-%Y', errors='coerce')

    # 3. Robust Merge
    # We join base_df with landsat, then with terra using INNER join to keep only matching rows
    print(f"Merging datasets... Initial base shape: {base_df.shape}")
    
    merged = pd.merge(base_df, landsat_df, on=['Latitude', 'Longitude', 'Sample Date'], how='inner')
    merged = pd.merge(merged, terra_df, on=['Latitude', 'Longitude', 'Sample Date'], how='inner')
    
    print(f"Merged shape: {merged.shape}")
    
    # 4. Feature Engineering: Extract seasonal features
    # SK-Learn cannot handle datetime objects directly, so we extract numeric features
    merged['Month'] = merged['Sample Date'].dt.month
    merged['DayOfYear'] = merged['Sample Date'].dt.dayofyear
    
    # Drop the original date column as it's not a numeric feature for the model
    # merged = merged.drop(columns=['Sample Date']) 
    # (Optional: Keep it for debugging, but don't pass it to the model)

    # 5. Handle Missing Values
    merged = merged.fillna(merged.median(numeric_only=True))
    
    return merged

def main():
    print("Loading data...")
    try:
        train_df = pd.read_csv(WQA_TRAIN)
        landsat_train = pd.read_csv(LANDSAT_TRAIN)
        terra_train = pd.read_csv(TERRA_TRAIN)

        val_df = pd.read_csv(SUB_TEMP)
        landsat_val = pd.read_csv(LANDSAT_VAL)
        terra_val = pd.read_csv(TERRA_VAL)
        
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    print("\nProcessing Training Data...")
    train_full_df = preprocess_and_merge(train_df, landsat_train, terra_train)
    
    print("\nProcessing Submission Data...")
    # For submission, we want to perform the same merge to get features for the rows in submission_template
    val_submission_df = preprocess_and_merge(val_df, landsat_val, terra_val, is_submission=True)

    if train_full_df.empty:
        print("CRITICAL ERROR: Training data is empty after merge! Check date formats and coordinate precision.")
        return
        
    if val_submission_df.empty:
        print("CRITICAL ERROR: Submission data is empty after merge!")
        return

    # Define targets and features
    targets = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
    features = ["nir", "green", "swir16", "swir22", "NDMI", "MNDWI", "pet", "Month", "DayOfYear"]

    available_features = [f for f in features if f in train_full_df.columns]
    print(f"\nUsing features for training: {available_features}")

    # Initial scaling based on full training data
    scaler = StandardScaler()
    X_full_raw = train_full_df[available_features]
    scaler.fit(X_full_raw)
    
    X_submission_scaled = scaler.transform(val_submission_df[available_features])

    # Create a copy of the merged submission df to store predictions
    # We use the merged one to ensure we have the rows corresponding to our predictions
    final_submission_rows = val_submission_df[['Latitude', 'Longitude', 'Sample Date']].copy()
    
    overall_r2 = []
    
    for target in targets:
        print(f"\n" + "="*50)
        print(f"MODEL TARGET: {target}")
        print("="*50)
        
        y_full = train_full_df[target]
        X_full_scaled = scaler.transform(X_full_raw)

        # 1. Hold-out Validation (20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X_full_scaled, y_full, test_size=0.2, random_state=42
        )
        
        # Train generic model for validation metrics
        val_model = HistGradientBoostingRegressor(
            max_iter=1000, learning_rate=0.05, max_depth=10, random_state=42
        )
        val_model.fit(X_train, y_train)
        y_pred_val = val_model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred_val)
        mae = mean_absolute_error(y_test, y_pred_val)
        print(f"Validation R2 Score: {r2:.4f}")
        print(f"Validation MAE:      {mae:.4f}")

        # 2. Final Training on Full Dataset
        print(f"Training final model on full dataset ({len(train_full_df)} rows)...")
        final_model = HistGradientBoostingRegressor(
            max_iter=2000, learning_rate=0.03, max_depth=12, random_state=42
        )
        final_model.fit(X_full_scaled, y_full)
        
        # Save model artifact
        target_sanitized = target.replace(" ", "_").lower()
        model_filename = os.path.join(MODEL_DIR, f"model_{target_sanitized}.joblib")
        joblib.dump(final_model, model_filename)
        
        # Predict on submission set
        preds = final_model.predict(X_submission_scaled)
        final_submission_rows[target] = preds
        
        overall_r2.append(r2)

    print(f"\n" + "="*50)
    print(f"Average R2 across targets: {np.mean(overall_r2):.4f}")

    # Format 'Sample Date' back to original string format for submission if needed, 
    # but usually submission requires specific format. Adjusting to string for CSV.
    # final_submission_rows['Sample Date'] = final_submission_rows['Sample Date'].dt.strftime('%d-%m-%Y')

    output_path = os.path.join(DATA_DIR, "improved_submission_v2.csv")
    final_submission_rows.to_csv(output_path, index=False)
    print(f"\nFinal submission saved to: {output_path}")

if __name__ == "__main__":
    main()
