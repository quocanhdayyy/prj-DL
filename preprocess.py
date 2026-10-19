"""Build reproducible, disjoint ordered-pair splits from SMILES CSV data."""
import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split

NUM_CLASSES = 86
N_BITS = 2048
COLUMNS = ['Smiles1', 'Smiles2', 'Label']


@lru_cache(maxsize=20000)
def canonical_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f'Missing SMILES: {smiles!r}')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles!r}')
    return Chem.MolToSmiles(mol, isomericSmiles=True)


@lru_cache(maxsize=20000)
def fingerprint(smiles):
    mol = Chem.MolFromSmiles(canonical_smiles(smiles))
    generator = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5, fpSize=N_BITS)
    result = np.zeros(N_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), result)
    return result


def pair_features(first, second):
    return np.concatenate([fingerprint(first), fingerprint(second)])


def clean_data(frame, reject_conflicts=True):
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    frame = frame[COLUMNS].copy()
    labels = pd.to_numeric(frame.Label, errors='raise')
    if labels.isna().any() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError('Labels must be integers in 1..86')
    if not labels.between(1, NUM_CLASSES).all():
        raise ValueError('Source labels must use 1..86 (not zero-based labels)')
    frame['Label'] = labels.astype(int)
    for column in COLUMNS[:2]:
        frame[column] = frame[column].map(canonical_smiles)
    # Keep drug order: interaction descriptions may be directional.
    conflicts = frame.groupby(COLUMNS[:2]).Label.nunique()
    if reject_conflicts and (conflicts > 1).any():
        raise ValueError('An ordered drug pair has conflicting labels; resolve source data first')
    return frame.drop_duplicates(COLUMNS).reset_index(drop=True)


def prepare(source, output, seed=42):
    original = pd.read_csv(source)
    frame = clean_data(original, reject_conflicts=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    conflict_mask = frame.groupby(COLUMNS[:2]).Label.transform('nunique') > 1
    conflicts = frame[conflict_mask]
    conflicts.to_csv(output / 'conflicting_pairs.csv', index=False)
    frame = frame[~conflict_mask].reset_index(drop=True)
    if set(frame.Label) != set(range(1, NUM_CLASSES + 1)):
        raise ValueError('Full source must contain all 86 classes')
    train, test = train_test_split(frame, test_size=0.2, stratify=frame.Label, random_state=seed)
    train, valid = train_test_split(train, test_size=0.2, stratify=train.Label, random_state=seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {'source': str(source), 'seed': seed, 'removed_duplicates': len(original)-len(frame)-len(conflicts),
               'quarantined_conflict_rows': len(conflicts),
               'fingerprint': {'type': 'RDKit', 'maxPath': 5, 'bits_per_drug': N_BITS},
               'source_label_base': 1, 'stored_label_base': 0,
               'split_protocol': 'stratified ordered canonical pair; drugs may recur across splits',
               'splits': {}}
    for name, part in [('train', train), ('valid', valid), ('test', test)]:
        part = part.reset_index(drop=True)
        X = np.stack([pair_features(a, b) for a, b in zip(part.Smiles1, part.Smiles2)])
        y = part.Label.to_numpy(dtype=np.int32) - 1
        np.savez_compressed(output / f'{name}_data.npz', X=X, y=y)
        part.to_csv(output / f'{name}.csv', index=False)
        summary['splits'][name] = {'samples': len(part), 'classes': int(part.Label.nunique())}
    (output / 'metadata.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).parent / 'SMILES_DDI_filtered.csv')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'data' / 'processed')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.source, args.output, args.seed)
