import pandas as pd
import numpy as np
import os

old_path = r'c:\Users\hp\Desktop\Ey challenge\Jupyter Notebook Package\processed sk one\submission.csv'
new_path = r'c:\Users\hp\Desktop\Ey challenge\submission.csv'

targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']

if not os.path.exists(old_path):
    print(f"Error: Old file not found at {old_path}")
    exit()
if not os.path.exists(new_path):
    print(f"Error: New file not found at {new_path}")
    exit()

old_df = pd.read_csv(old_path)
new_df = pd.read_csv(new_path)

print("-" * 80)
print(f"{'Target Variable':<35} | {'Metric':<10} | {'Old Sub':<10} | {'New Sub (XGB)':<10}")
print("-" * 80)

for t in targets:
    corr = old_df[t].corr(new_df[t])
    print(f"{t:<35} | Mean   | {old_df[t].mean():<10.2f} | {new_df[t].mean():<10.2f}")
    print(f"{'':<35} | Std    | {old_df[t].std():<10.2f} | {new_df[t].std():<10.2f}")
    print(f"{'':<35} | Max    | {old_df[t].max():<10.2f} | {new_df[t].max():<10.2f}")
    print(f"{'':<35} | Correlation: {corr:.4f}")
    print("-" * 80)
