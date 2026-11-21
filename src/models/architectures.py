"""Architecture registry; previous SNN+MLP remains in src.models.siamese."""
import tensorflow as tf
from src.models.siamese import L2Normalize

ARCHITECTURES = {
    'mlp', 'snn', 'prototype', 'snn_mlp', 'snn_prototype', 'mlp_prototype'
}

def architecture_name(cfg):
    name = str(cfg.get('architecture', {}).get('name', 'snn_prototype')).lower()
    if name not in ARCHITECTURES:
        raise ValueError(f'Unknown architecture {name!r}. Available: {sorted(ARCHITECTURES)}')
    return name


def shared_encoder(input_dim, model_cfg, prefix='encoder'):
    reg = tf.keras.regularizers.l2(float(model_cfg.get('l2_regularization', 0.0001)))
    inp = tf.keras.Input(shape=(input_dim,), name=f'{prefix}_input')
    x = inp
    for n, units in enumerate(model_cfg.get('projection_dims', [1024, 256]), 1):
        x = tf.keras.layers.Dense(int(units), kernel_regularizer=reg, name=f'{prefix}_dense{n}')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{prefix}_bn{n}')(x)
        x = tf.keras.layers.ReLU(name=f'{prefix}_relu{n}')(x)
        x = tf.keras.layers.Dropout(float(model_cfg.get('dropout', .3)), name=f'{prefix}_drop{n}')(x)
    out = L2Normalize(name=f'{prefix}_l2')(x)
    return tf.keras.Model(inp, out, name=f'{prefix}_model')


def build_mlp(input_dim, class_count, cfg, lr):
    """Supervised MLP; feature extractor also supports MLP -> prototype."""
    encoder = shared_encoder(input_dim, cfg, prefix='mlp')
    inp = tf.keras.Input(shape=(input_dim,), name='embedding')
    z = encoder(inp)
    pred = tf.keras.layers.Dense(class_count, activation='softmax', name='relation_class')(z)
    model = tf.keras.Model(inp, pred, name='supervised_mlp')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, encoder


def build_snn_cosine(input_dim, cfg, lr):
    """SNN without decision-head MLP: shared encoder + symmetric cosine sigmoid score."""
    encoder = shared_encoder(input_dim, cfg, prefix='cosine')
    left = tf.keras.Input((input_dim,), name='left_embedding')
    right = tf.keras.Input((input_dim,), name='right_embedding')
    a, b = encoder(left), encoder(right)
    cos = tf.keras.layers.Dot(axes=1, normalize=True, name='cosine_similarity')([a, b])
    # Scale from [-1,1] to [0,1], keeping the similarity function symmetric.
    score = tf.keras.layers.Rescaling(scale=.5, offset=.5, name='pair_score')(cos)
    model = tf.keras.Model([left, right], score, name='siamese_cosine')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model, encoder
