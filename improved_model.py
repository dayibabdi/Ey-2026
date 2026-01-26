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

def preprocess_data(df, landsat_df, terra_df):
    # Combine datasets vertically
    combined = pd.concat([df, landsat_df, terra_df], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    
    # Convert 'Sample Date' to datetime
    combined['Sample Date'] = pd.to_datetime(combined['Sample Date'], format='%d-%m-%Y')
    
    # Feature Engineering: Extract seasonal features
    combined['Month'] = combined['Sample Date'].dt.month
    combined['DayOfYear'] = combined['Sample Date'].dt.dayofyear
    
    # Fill missing values with median
    combined = combined.fillna(combined.median(numeric_only=True))
    
    return combined

def main():
    print("Loading data...")
    try:
        train_df = pd.read_csv(WQA_TRAIN)
        landsat_train = pd.read_csv(LANDSAT_TRAIN)
        terra_train = pd.read_csv(TERRA_TRAIN)

        val_df = pd.read_csv(SUB_TEMP)
        landsat_val = pd.read_csv(LANDSAT_VAL)
        terra_val = pd.read_csv(TERRA_VAL)
        
        print(f"Total training data size: {len(train_df)} rows")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    print("Preprocessing data...")
    train_full_df = preprocess_data(train_df, landsat_train, terra_train)
    val_submission_df = preprocess_data(val_df, landsat_val, terra_val)

    # Define targets and features
    targets = ["Total Alkalinity", "Electrical Conductance", "Dissolved Reactive Phosphorus"]
    features = ["nir", "green", "swir16", "swir22", "NDMI", "MNDWI", "pet", "Month", "DayOfYear"]

    available_features = [f for f in features if f in train_full_df.columns]
    print(f"Using features: {available_features}")

    # Initial scaling based on full training data
    scaler = StandardScaler()
    X_full_raw = train_full_df[available_features]
    scaler.fit(X_full_raw)
    
    X_submission_scaled = scaler.transform(val_submission_df[available_features])

    submission = val_df.copy()
    overall_r2 = []
    
    for target in targets:
        print(f"\n" + "="*50)
        print(f"EVALUATING MODEL FOR: {target}")
        print("="*50)
        
        y_full = train_full_df[target]
        X_full_scaled = scaler.transform(X_full_raw)

        # 1. Hold-out Validation (20%) to show "Digits" comparison
        X_train, X_test, y_train, y_test = train_test_split(
            X_full_scaled, y_full, test_size=0.2, random_state=42
        )
        
        test_model = HistGradientBoostingRegressor(
            max_iter=1000, learning_rate=0.05, max_depth=10, random_state=42
        )
        test_model.fit(X_train, y_train)
        
        # Predictions on hold-out set
        y_pred = test_model.predict(X_test)
        
        # Performance Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\nHold-out Performance (20% of training data):")
        print(f"  - R2 Score:  {r2:.4f}")
        print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
        
        # Print Sample "Digits" (Actual vs Predicted)
        print(f"\nSample Predictions (First 10 values of hold-out set):")
        comparison = pd.DataFrame({
            'Actual Digit': y_test.values[:10],
            'Predicted Digit': y_pred[:10],
            'Difference': np.abs(y_test.values[:10] - y_pred[:10])
        })
        print(comparison.to_string(index=False))

        # 2. Final Training on Full Dataset for Submission
        print(f"\nTraining final candidate model on 100% of data...")
        final_model = HistGradientBoostingRegressor(
            max_iter=2000, learning_rate=0.03, max_depth=12, random_state=42
        )
        final_model.fit(X_full_scaled, y_full)
        
        # Save bundle
        target_sanitized = target.replace(" ", "_").lower()
        model_filename = os.path.join(MODEL_DIR, f"model_bundle_{target_sanitized}.joblib")
        joblib.dump({"model": final_model, "scaler": scaler, "features": available_features}, model_filename)
        
        # Predict for actual challenge submission
        submission[target] = final_model.predict(X_submission_scaled)
        
        overall_r2.append(r2)

    print(f"\n\nFinal Summary Performance:")
    print(f"  Average Hold-out R2 across all targets: {np.mean(overall_r2):.4f}")

    output_path = os.path.join(DATA_DIR, "improved_submission_v2.csv")
    submission.to_csv(output_path, index=False)
    print(f"\nFinal submission saved to: {output_path}")

if __name__ == "__main__":
    main()
