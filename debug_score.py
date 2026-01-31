import pandas as pd
import numpy as np

# Check for raw duplicates
print("\n--- RAW DATA DUPLICATE CHECK (Lat, Lon, Date) ---")
wqa = pd.read_csv(r'c:\Users\hp\Desktop\Ey challenge\Jupyter Notebook Package\Jupyter Notebook Package\water_quality_training_dataset.csv')
land = pd.read_csv(r'c:\Users\hp\Desktop\Ey challenge\Jupyter Notebook Package\Jupyter Notebook Package\landsat_features_training.csv')
terra = pd.read_csv(r'c:\Users\hp\Desktop\Ey challenge\Jupyter Notebook Package\Jupyter Notebook Package\terraclimate_features_training.csv')

def count_dupes(df, name):
    # Strip whitespace from columns
    df.columns = df.columns.str.strip()
    dupes = df.duplicated(subset=['Latitude', 'Longitude', 'Sample Date']).sum()
    print(f"{name:<15}: {len(df)} rows, {dupes} duplicates")

count_dupes(wqa, "Water Quality")
count_dupes(land, "Landsat")
count_dupes(terra, "TerraClimate")

print(f"\n--- COORDINATE PRECISION (WQA sample) ---")
print(wqa[['Latitude', 'Longitude']].head(3))
print(f"\n--- COORDINATE PRECISION (Landsat sample) ---")
print(land[['Latitude', 'Longitude']].head(3))

# Check for precision issues (round to 6 decimal places and check duplicates again)
print("\n--- PRECISION SENSITIVITY CHECK (Rounded to 5 decimals) ---")
for df, name in [(wqa, "WQA"), (land, "Landsat"), (terra, "Terra")]:
    df_c = df.copy()
    df_c['Latitude'] = df_c['Latitude'].round(5)
    df_c['Longitude'] = df_c['Longitude'].round(5)
    dupes = df_c.duplicated(subset=['Latitude', 'Longitude', 'Sample Date']).sum()
    print(f"{name:<15}: {dupes} duplicates when rounded")
