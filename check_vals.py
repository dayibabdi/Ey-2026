import pandas as pd
import os

old_path = r'c:\Users\hp\Desktop\Ey challenge\Jupyter Notebook Package\processed sk one\submission.csv'
new_path = r'c:\Users\hp\Desktop\Ey challenge\submission.csv'

old = pd.read_csv(old_path)
new = pd.read_csv(new_path)

print("--- ALKALINITY ---")
print("OLD:", old['Total Alkalinity'].head().tolist())
print("NEW:", new['Total Alkalinity'].head().tolist())

print("\n--- CONDUCTANCE ---")
print("OLD:", old['Electrical Conductance'].head().tolist())
print("NEW:", new['Electrical Conductance'].head().tolist())
