import os

import pandas as pd


def load_raw_data() -> pd.DataFrame:
    """
    Load data.csv file located in data.raw folder
    """
    # Load and return the CSV file
    return pd.read_csv("data/raw/data.csv")


def load_prepared_data(base_dir: str) -> pd.DataFrame:
    """
    Load prepared_data.parquet file located in data/prepared folder.
    """
    file_path = os.path.join(base_dir, "data/prepared/prepared_data.parquet")
    return pd.read_parquet(file_path)
