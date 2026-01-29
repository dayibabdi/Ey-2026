import pandas as pd
import os

files = [
    'Jupyter Notebook Package/Jupyter Notebook Package/water_quality_training_dataset.csv',
    'Jupyter Notebook Package/Jupyter Notebook Package/terraclimate_features_training.csv',
    'Jupyter Notebook Package/Jupyter Notebook Package/landsat_features_training.csv',
    'Jupyter Notebook Package/Jupyter Notebook Package/submission_template.csv'
]

with open('data_analysis.txt', 'w') as out:
    for f in files:
        out.write(f"\n{'='*20}\nFile: {f}\n")
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                out.write(f"Shape: {df.shape}\n")
                out.write(f"Columns: {list(df.columns)}\n")
                out.write(f"Head:\n{df.head().to_string()}\n")
            except Exception as e:
                out.write(f"Error reading: {e}\n")
        else:
            out.write("File not found.\n")
