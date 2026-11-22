from pathlib import Path
import pandas as pd
import numpy as np


def load_table(path, label_column, adapter=None, column_map=None):
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported dataset extension: {ext}")
    if column_map:
        if not isinstance(column_map, dict):
            raise ValueError('dataset.column_map must be a mapping: original_column -> standard_column')
        missing = sorted(set(column_map) - set(df.columns))
        if missing:
            raise ValueError(f'Missing columns required by dataset.column_map: {missing}')
        if len(set(column_map.values())) != len(column_map.values()):
            raise ValueError('dataset.column_map cannot target the same column twice')
        df = df.rename(columns=column_map)
    if label_column not in df:
        raise ValueError(f"Missing required label column: {label_column}")
    if adapter == 'core':
        required = ('_source_split', 'context', 'e1_name', 'e2_name', 'invert_relation', 'source_id')
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f'CORE dataset requires normalized columns {missing}. Run python -m src.data.prepare_core first.')
        if df['source_id'].isna().any() or df['source_id'].duplicated().any():
            raise ValueError('CORE source_id must be present and unique')
        import hashlib
        import json
        manifest_path = Path(path).with_suffix('.manifest.json')
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if digest != manifest.get('sha256'):
                raise ValueError('CORE file differs from its source manifest; do not trust cached embeddings')
    df = df.reset_index(drop=True)
    df["_original_row_id"] = np.arange(len(df), dtype=np.int64)
    return df
