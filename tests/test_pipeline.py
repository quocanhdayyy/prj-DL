import numpy as np
import pandas as pd
import pytest
from preprocess import clean_data, pair_features
from model import create_model, load_data


def test_invalid_smiles_rejected():
    with pytest.raises(ValueError, match='Invalid SMILES'):
        pair_features('not-a-molecule', 'CCO')


def test_canonical_duplicates_and_conflicting_labels():
    frame = pd.DataFrame({'Smiles1': ['CCO', 'OCC'], 'Smiles2': ['CC', 'CC'], 'Label': [1, 1]})
    assert len(clean_data(frame)) == 1
    frame.loc[1, 'Label'] = 2
    with pytest.raises(ValueError, match='conflicting'):
        clean_data(frame)


def test_pair_order_preserved():
    x = pair_features('CCO', 'c1ccccc1')
    reverse = pair_features('c1ccccc1', 'CCO')
    assert x.shape == (4096,)
    assert x.dtype == np.uint8
    np.testing.assert_array_equal(x[:2048], reverse[2048:])
    assert not np.array_equal(x, reverse)


def test_bad_labels_rejected(tmp_path):
    path = tmp_path / 'bad.npz'
    np.savez(path, X=np.zeros((2, 4096)), y=np.array([1, 86]))
    with pytest.raises(ValueError, match='zero-based'):
        load_data(path)


@pytest.mark.parametrize('architecture', ['mlp', 'cnn-bilstm'])
def test_model_roundtrip(architecture, tmp_path):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    model = create_model(architecture)
    x = np.zeros((2, 4096, 1) if architecture == 'cnn-bilstm' else (2, 4096), dtype=np.float32)
    scores = model(x, training=False).numpy()
    assert scores.shape == (2, 86)
    np.testing.assert_allclose(scores.sum(axis=1), 1, atol=1e-5)
    path = tmp_path / 'model.keras'
    model.save(path)
    np.testing.assert_allclose(tf.keras.models.load_model(path)(x, training=False), scores, atol=1e-6)


def test_prepare_splits_are_disjoint_and_quarantine_conflicts(tmp_path):
    import json
    from preprocess import prepare
    rows = []
    for i in range(860):
        rows.append({'Smiles1': 'C' * (i // 30 + 1), 'Smiles2': 'C' * (i % 30 + 1), 'Label': i % 86 + 1})
    rows += [rows[0].copy(), dict(rows[0], Label=2)]
    source = tmp_path / 'source.csv'
    pd.DataFrame(rows).to_csv(source, index=False)
    output = tmp_path / 'processed'
    prepare(source, output)
    parts = [pd.read_csv(output / f'{name}.csv') for name in ['train', 'valid', 'test']]
    keys = [set(zip(p.Smiles1, p.Smiles2)) for p in parts]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    assert all(p.Label.nunique() == 86 for p in parts)
    assert json.loads((output / 'metadata.json').read_text())['quarantined_conflict_rows'] == 2


def test_smoke_sampling_keeps_singleton_classes(tmp_path):
    y = np.array([0] * 30 + [1, 2])
    X = np.zeros((len(y), 4096), dtype=np.uint8)
    path = tmp_path / 'data.npz'
    np.savez(path, X=X, y=y)
    _, sampled = load_data(path, limit=10)
    assert set(sampled) == {0, 1, 2}
    np.testing.assert_array_equal(sampled, load_data(path, limit=10)[1])
