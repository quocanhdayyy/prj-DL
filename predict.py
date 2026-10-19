"""Predict a DDI class for an ordered pair of SMILES."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from preprocess import pair_features


def predict(run, smiles1, smiles2, top_k=5):
    if not 1 <= top_k <= 86:
        raise ValueError('top-k must be in 1..86')
    config = json.loads((run / 'config.json').read_text())
    features = pair_features(smiles1, smiles2).astype(np.float32)[None, :]
    if config['architecture'] == 'cnn-bilstm':
        features = features[..., None]
    model = tf.keras.models.load_model(run / 'best_model.keras')
    scores = model(features, training=False).numpy()[0]
    return {'smoke_model': config['smoke_only'], 'predictions': [
        {'label': int(i)+1, 'probability': float(scores[i])}
        for i in np.argsort(scores)[::-1][:top_k]]}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--smiles1', required=True)
    p.add_argument('--smiles2', required=True)
    p.add_argument('--top-k', type=int, default=5)
    a = p.parse_args()
    print(json.dumps(predict(a.run, a.smiles1, a.smiles2, a.top_k), indent=2))
