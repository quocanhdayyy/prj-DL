"""Audit generated splits, class balance and canonical ordered-pair leakage."""
import argparse
from pathlib import Path
import pandas as pd
from model import load_data
from preprocess import canonical_smiles


def audit(directory):
    pairs = {}
    for name in ['train', 'valid', 'test']:
        X, y = load_data(directory / f'{name}_data.npz')
        frame = pd.read_csv(directory / f'{name}.csv')
        if len(frame) != len(y) or not (frame.Label.to_numpy() - 1 == y).all():
            raise ValueError(f'{name}: CSV/NPZ labels differ')
        pairs[name] = set(zip(frame.Smiles1.map(canonical_smiles), frame.Smiles2.map(canonical_smiles)))
        if len(pairs[name]) != len(frame):
            raise ValueError(f'{name}: duplicate pair')
        counts = frame.Label.value_counts()
        if set(counts.index) != set(range(1, 87)):
            raise ValueError(f'{name}: missing classes')
        print(f'{name}: X={X.shape}, classes={len(counts)}, smallest={counts.min()}, largest={counts.max()}')
    for a, b in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        overlap = len(pairs[a] & pairs[b])
        print(f'{a}/{b}: {overlap} overlapping canonical ordered pairs')
        if overlap:
            raise ValueError('Split leakage detected')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path(__file__).parent / 'data' / 'processed')
    audit(p.parse_args().data_dir)
