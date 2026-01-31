import pandas as pd
import numpy as np

train = pd.read_csv('processed_data/train_final.csv')
test = pd.read_csv('processed_data/test_final.csv')

features = ["Latitude", "Longitude", "nir", "green", "swir16", "swir22", "NDMI", "MNDWI", "pet"]

print("--- NaN PERCENTAGE ---")
for f in features:
    tr_nan = train[f].isnull().mean() * 100
    ts_nan = test[f].isnull().mean() * 100
    print(f"{f:<10} | Train: {tr_nan:>6.2f}% | Test: {ts_nan:>6.2f}%")

print("\n--- TARGET STATS (ALKALINITY) ---")
print(train['Total Alkalinity'].describe())
