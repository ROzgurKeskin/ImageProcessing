import pandas as pd
from pathlib import Path
import random

def get_csv_path():
    return r"C:\PythonProject\2.donem projeleri\SayisalGoruntu\SkinCancerISIC\train_df.csv"

def load_dataframe(csv_path=None):
    if csv_path is None:
        csv_path = get_csv_path()
    return pd.read_csv(csv_path)

def sample_images(df, n=10, random_state=42):
    return df.sample(n=n, random_state=random_state)
