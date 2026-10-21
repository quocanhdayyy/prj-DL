import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import model as model_module
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


def test_skip_test_writes_validation_metrics_without_loading_test(tmp_path, monkeypatch):
    calls = []
    y = np.array([0, 1], dtype=np.int32)
    x = np.zeros((2, 4096), dtype=np.uint8)
    monkeypatch.setattr(model_module, 'load_data', lambda path, limit, seed: (calls.append(path.name) or (x, y)))
    monkeypatch.setattr(model_module, 'dataset', lambda X, y, *args: (X, y))
    monkeypatch.setattr(model_module, 'classification_report', lambda *a, **k: {
        'macro avg': {'f1-score': 0.25}, 'weighted avg': {'f1-score': 0.25}})
    monkeypatch.setattr(model_module, 'accuracy_score', lambda *a: 0.5)

    class FakeModel:
        def evaluate(self, data, verbose=0):
            return 1.0, 0.5

        def predict(self, data, verbose=0):
            return np.zeros((2, 86))

    args = SimpleNamespace(data_dir=tmp_path, output=tmp_path / 'run', architecture='mlp',
                           batch_size=8, seed=42, smoke=True, skip_test=True)
    args.output.mkdir()
    model_module.save_training_evaluations(args, FakeModel())

    assert calls == ['valid_data.npz']
    metrics = json.loads((args.output / 'validation_metrics.json').read_text())
    assert metrics['macro_f1'] == 0.25 and metrics['smoke_only'] is True
    assert not (args.output / 'metrics.json').exists()


def test_evaluate_only_uses_run_config_and_refuses_repeat(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'config.json').write_text(json.dumps({
        'data_dir': str(tmp_path / 'data'), 'architecture': 'mlp', 'seed': 17,
        'batch_size': 7, 'smoke_only': True}))
    loaded = []
    class FakeModel:
        pass
    monkeypatch.setattr(model_module.tf.keras.models, 'load_model', lambda path: (loaded.append(path) or FakeModel()))
    evaluated = []
    monkeypatch.setattr(model_module, 'evaluate_split', lambda model, data_dir, split, architecture, batch_size, seed, smoke:
        (evaluated.append((data_dir, split, architecture, batch_size, seed, smoke)) or
         ({'loss': 1.0, 'accuracy': 0.5, 'macro_f1': 0.25, 'weighted_f1': 0.25,
           'micro_f1': 0.5, 'smoke_only': smoke, 'classification_report': {}},
          np.array([0]), np.array([0]))))
    args = SimpleNamespace(output=run)
    model_module.evaluate_only(args)
    assert evaluated == [(tmp_path / 'data', 'test', 'mlp', 7, 17, True)]
    assert loaded == [run / 'best_model.keras']
    assert json.loads((run / 'metrics.json').read_text())['smoke_only'] is True

    with pytest.raises(FileExistsError, match='refusing to evaluate test data again'):
        model_module.evaluate_only(args)
    assert len(evaluated) == 1


@pytest.mark.parametrize('artifact', ['metrics.json', 'confusion_matrix.npy', 'test_predictions.npz'])
def test_evaluate_only_refuses_partial_test_artifacts(tmp_path, artifact):
    run = tmp_path / 'run'
    run.mkdir()
    (run / artifact).touch()
    with pytest.raises(FileExistsError, match='refusing to evaluate test data again'):
        model_module.evaluate_only(SimpleNamespace(output=run))
