from pathlib import Path
import pandas as pd
import numpy as np


def load_table(path, label_column):
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset extension: {ext}")
    if label_column not in df:
        raise ValueError(f"Missing required label column: {label_column}")
    df = df.reset_index(drop=True)
    df["_original_row_id"] = np.arange(len(df), dtype=np.int64)
    return df
