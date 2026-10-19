"""Train CNN-BiLSTM or MLP on validated DDI fingerprints."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Dropout, BatchNormalization, MaxPooling1D, Dense, LSTM, Bidirectional, Activation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

NUM_CLASSES = 86
ROOT = Path(__file__).resolve().parent

def build_model(input_shape, num_classes):

    model = Sequential([

        Input(shape=input_shape),

        # -------------------------
        # CNN Block 1
        # -------------------------
        Conv1D(
            filters=16,
            kernel_size=3,
            activation="relu"
        ),

        BatchNormalization(),

        MaxPooling1D(
            pool_size=2,
            strides=2
        ),

        Dropout(0.1),

        # -------------------------
        # CNN Block 2
        # -------------------------
        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu"
        ),

        BatchNormalization(),

        MaxPooling1D(
            pool_size=2
        ),

        Dropout(0.1),

        # -------------------------
        # CNN Block 3
        # -------------------------
        Conv1D(
            filters=48,
            kernel_size=3,
            activation="relu"
        ),

        BatchNormalization(),

        MaxPooling1D(
            pool_size=2
        ),

        Dropout(0.2),

        # -------------------------
        # CNN Block 4
        # -------------------------
        Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu"
        ),

        BatchNormalization(),

        # -------------------------
        # BiLSTM
        # -------------------------
        Bidirectional(
            LSTM(
                128,
                return_sequences=True
            )
        ),

        Activation("relu"),

        Bidirectional(
            LSTM(96)
        ),

        # -------------------------
        # Fully Connected
        # -------------------------
        Dense(
            256,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),

        # 86-class classification
        Dense(
            num_classes,
            activation="softmax"
        )
    ])

    return model


def create_model(architecture):
    if architecture == 'cnn-bilstm':
        return build_model((4096, 1), NUM_CLASSES)
    return Sequential([
        Input((4096,)), Dense(512, activation='relu'), Dropout(0.3),
        Dense(256, activation='relu'), Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ], name='fingerprint_mlp')


def load_data(path, limit=None, seed=42):
    with np.load(path, allow_pickle=False) as data:
        X, y = data['X'], data['y']
    if X.ndim != 2 or X.shape[1] != 4096 or len(X) != len(y) or len(y) == 0:
        raise ValueError(f'{path}: expected nonempty X=(N,4096), y=(N,)')
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= NUM_CLASSES:
        raise ValueError(f'{path}: labels must be zero-based integers 0..85')
    if not np.isin(X, [0, 1]).all():
        raise ValueError(f'{path}: fingerprints must be binary')
    if limit and len(y) > limit:
        rng = np.random.default_rng(seed)
        classes = np.unique(y)
        if limit < len(classes):
            raise ValueError('Sample limit must cover every observed class')
        mandatory = np.array([rng.choice(np.flatnonzero(y == cls)) for cls in classes])
        remaining = np.setdiff1d(np.arange(len(y)), mandatory)
        indices = np.concatenate([mandatory, rng.choice(remaining, limit-len(mandatory), replace=False)])
        rng.shuffle(indices)
        X, y = X[indices], y[indices]
    return X.astype(np.uint8, copy=False), y


def dataset(X, y, architecture, batch_size, training=False, seed=42):
    if architecture == 'cnn-bilstm':
        X = X[..., None]
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(min(len(y), 10000), seed=seed)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return ds.batch(batch_size).map(lambda x, label: (tf.cast(x, tf.float32), label)).with_options(options).prefetch(1)


def train(args):
    if args.learning_rate <= 0:
        raise ValueError('learning-rate must be positive')
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError('epochs and batch-size must be positive')
    if not (args.data_dir / 'metadata.json').exists():
        raise ValueError('Run preprocess.py first; legacy root NPZ files have split leakage')
    args.output.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(args.seed)
    if args.deterministic:
        tf.config.experimental.enable_op_determinism()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(tensorflow=tf.__version__, num_classes=NUM_CLASSES,
                  source_label_base=1, smoke_only=args.smoke)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    (args.output / 'data_metadata.json').write_text((args.data_dir / 'metadata.json').read_text())
    X, y = load_data(args.data_dir / 'train_data.npz', 1000 if args.smoke else None, args.seed)
    V, vy = load_data(args.data_dir / 'valid_data.npz', 300 if args.smoke else None, args.seed)
    classes = np.unique(y)
    if len(classes) != NUM_CLASSES:
        raise ValueError('Training set must contain all 86 classes')
    weights = dict(zip(classes.tolist(), compute_class_weight('balanced', classes=classes, y=y).tolist()))
    model = create_model(args.architecture)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    checkpoint = args.output / 'best_model.keras'
    history = model.fit(dataset(X, y, args.architecture, args.batch_size, True, args.seed),
        validation_data=dataset(V, vy, args.architecture, args.batch_size),
        epochs=1 if args.smoke else args.epochs, class_weight=weights, callbacks=[
            tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_loss', save_best_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6),
            tf.keras.callbacks.CSVLogger(str(args.output / 'training.csv')),
            tf.keras.callbacks.TerminateOnNaN()])
    (args.output / 'history.json').write_text(json.dumps(history.history, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ['loss', 'accuracy']):
        ax.plot(history.history[metric], label='train')
        ax.plot(history.history['val_' + metric], label='validation')
        ax.set(xlabel='Epoch (zero-based)', ylabel=metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'training_history.png')
    plt.close(fig)
    del X, y, V, vy
    model = tf.keras.models.load_model(checkpoint)
    X, y = load_data(args.data_dir / 'test_data.npz', 300 if args.smoke else None, args.seed)
    test = dataset(X, y, args.architecture, args.batch_size)
    loss, accuracy = model.evaluate(test, verbose=0)
    predicted = model.predict(test, verbose=0).argmax(axis=1)
    report = classification_report(y, predicted, labels=list(range(NUM_CLASSES)),
        target_names=[str(i) for i in range(1, NUM_CLASSES+1)], output_dict=True, zero_division=0)
    results = {'test_loss': float(loss), 'accuracy': float(accuracy),
               'macro_f1': report['macro avg']['f1-score'], 'weighted_f1': report['weighted avg']['f1-score'],
               'micro_f1': float(accuracy_score(y, predicted)), 'smoke_only': args.smoke,
               'classification_report': report}
    (args.output / 'metrics.json').write_text(json.dumps(results, indent=2))
    np.save(args.output / 'confusion_matrix.npy', confusion_matrix(y, predicted, labels=range(NUM_CLASSES)))
    np.savez_compressed(args.output / 'test_predictions.npz', y_true=y, y_pred=predicted)
    print(json.dumps({k:v for k,v in results.items() if k != 'classification_report'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'processed')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs' / 'cnn-bilstm')
    parser.add_argument('--architecture', choices=['cnn-bilstm', 'mlp'], default='cnn-bilstm')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.000173)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--deterministic', action='store_true')
    train(parser.parse_args())
