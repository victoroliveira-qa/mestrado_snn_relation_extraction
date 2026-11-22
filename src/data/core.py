"""CORE (Borchert et al., EMNLP 2023) data adapter.

Official 'train' and 'test' partitions are kept separate. No labels are
translated into Business labels; cross-domain comparisons require an explicit
label ontology and evaluation protocol.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

REQUIRED = ('context', 'relation', 'e1_name', 'e2_name', 'invert_relation')
SPLITS = ('train', 'test')


def _text(value, name, row_number):
    if isinstance(value, list):
        if not all(isinstance(token, str) for token in value):
            raise ValueError(f'CORE row {row_number}: {name} tokens must be strings')
        return ' '.join(value)
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f'CORE row {row_number}: missing/invalid {name}')


def canonical_core_record(row, split, row_number):
    missing = [k for k in REQUIRED if k not in row]
    if missing:
        raise ValueError(f'CORE row {row_number}: missing fields {missing}')
    context = _text(row['context'], 'context', row_number)
    relation = str(row['relation']).strip()
    if not relation:
        raise ValueError(f'CORE row {row_number}: relation is empty')
    e1 = _text(row['e1_name'], 'e1_name', row_number)
    e2 = _text(row['e2_name'], 'e2_name', row_number)
    inverse = row['invert_relation']
    if isinstance(inverse, bool):
        inverse = int(inverse)
    if inverse not in (0, 1):
        raise ValueError(f'CORE row {row_number}: invert_relation must be 0 or 1')
    # Preserve the original directional metadata; DO NOT silently swap entities.
    stable = f'{split}:{row.get("id", row_number)}'
    return {
        'source_id': stable,
        '_source_split': split,
        'relation_label': relation,
        'context': context,
        'e1_name': e1,
        'e2_name': e2,
        'invert_relation': int(inverse),
        'e1_start': row.get('e1_start'), 'e1_end': row.get('e1_end'),
        'e2_start': row.get('e2_start'), 'e2_end': row.get('e2_end'),
        'embedding_text': f'Entity 1: {e1}\nEntity 2: {e2}\nDirection inverted: {bool(inverse)}\nContext: {context}',
    }


def normalize_core_partitions(train_rows, test_rows):
    """Normalize official split records without filtering classes or rows."""
    if not train_rows or not test_rows:
        raise ValueError('Both official CORE partitions must contain records')
    normalized = []
    for split, rows in (('train', train_rows), ('test', test_rows)):
        for index, row in enumerate(rows):
            normalized.append(canonical_core_record(row, split, index))
    df = pd.DataFrame(normalized)
    if df['source_id'].duplicated().any():
        raise ValueError('Duplicated source_id in CORE partitions; cannot trace records safely')
    return df


def prepare_core_from_huggingface(output_path, repo='pborchert/CORE', revision=None):
    """Fetch the official dataset using datasets, then persist a stable CSV + audit manifest.

    Requires optional 'datasets' dependency and network access at execution.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError('Install CORE dependency: pip install -r requirements-datasets.txt') from exc
    kwargs = {'revision': revision} if revision else {}
    ds = load_dataset(repo, **kwargs)
    for split in SPLITS:
        if split not in ds:
            raise ValueError(f'Dataset {repo} does not provide the official {split} split')
    df = normalize_core_partitions(list(ds['train']), list(ds['test']))
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.with_suffix('.manifest.json').exists():
        raise FileExistsError(f'CORE outputs already exist at {dest}; use a different path to preserve provenance')
    df.to_csv(dest, index=False)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    manifest = {
        'source_repo': repo, 'revision': revision,
        'split_counts': df['_source_split'].value_counts().to_dict(),
        'num_relations': int(df['relation_label'].nunique()),
        'row_count': int(len(df)), 'sha256': digest,
        'output': str(dest),
        'protocol_note': 'Official train/test partitions preserved; validation must come from train only. Not equivalent to the paper few-shot domain-adaptation protocol.',
    }
    dest.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return dest, manifest
