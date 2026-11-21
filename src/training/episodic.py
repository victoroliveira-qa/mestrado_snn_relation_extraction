"""True episodic N-way K-shot prototypical training, not simply centroid inference."""
import numpy as np
import tensorflow as tf


def sample_episode(X, y, rng, n_way, k_shot, queries):
    by_class = {int(c): np.flatnonzero(y == c) for c in np.unique(y)}
    eligible = [c for c, ids in by_class.items() if len(ids) >= k_shot + queries]
    if len(eligible) < n_way:
        raise ValueError(f'Episodic training requires {n_way} classes with at least {k_shot + queries} train examples each; found {len(eligible)}')
    classes = rng.choice(eligible, n_way, replace=False)
    sids, qids, qy = [], [], []
    for local_id, c in enumerate(classes):
        ids = rng.choice(by_class[int(c)], k_shot + queries, replace=False)
        sids.extend(ids[:k_shot]); qids.extend(ids[k_shot:]); qy.extend([local_id] * queries)
    return X[sids], X[qids], np.asarray(qy, dtype=np.int32)


def train_prototypical(encoder, X_train, y_train, X_val, y_val, config, output):
    """Episode-based metric learning; validation compares held-out points with TRAIN prototypes."""
    from src.evaluation.prototypes import compute_prototypes
    cfg = config['training']; ecfg = config.get('prototypical', {})
    n_way = int(ecfg.get('n_way', 3)); k = int(ecfg.get('k_shot', 2)); q = int(ecfg.get('queries', 2))
    steps = int(cfg.get('steps_per_epoch', 100)); temperature = float(ecfg.get('temperature', 10.0))
    if not (n_way >= 2 and k >= 1 and q >= 1 and temperature > 0):
        raise ValueError('n_way>=2, k_shot>=1, queries>=1 and temperature>0 required')
    rng = np.random.default_rng(int(config['experiment']['seed']))
    optimizer = tf.keras.optimizers.Adam(float(cfg['learning_rate']))
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    best_weights = None; best = float('inf'); no_improve = 0
    patience = int(cfg.get('early_stopping_patience', 10))
    num_cls = int(np.max(y_train)) + 1 if len(y_train) else 0
    for epoch in range(int(cfg['epochs'])):
        losses = []; accuracies = []
        for _ in range(steps):
            support, query, query_y = sample_episode(X_train, y_train, rng, n_way, k, q)
            with tf.GradientTape() as tape:
                zs = encoder(support, training=True)
                zq = encoder(query, training=True)
                proto = tf.reduce_mean(tf.reshape(zs, (n_way, k, -1)), axis=1)
                proto = tf.math.l2_normalize(proto, axis=1)
                logits = temperature * tf.matmul(zq, proto, transpose_b=True)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(query_y, logits, from_logits=True))
                if encoder.losses:
                    loss += tf.add_n(encoder.losses)
            grads = tape.gradient(loss, encoder.trainable_variables)
            optimizer.apply_gradients([(g,v) for g,v in zip(grads, encoder.trainable_variables) if g is not None])
            losses.append(float(loss.numpy()))
            accuracies.append(float(np.mean(tf.argmax(logits, axis=1).numpy() == query_y)))
        # Validation uses TRAIN prototypes exclusively, not test or validation examples.
        tr_proj = encoder.predict(X_train, batch_size=256, verbose=0)
        va_proj = encoder.predict(X_val, batch_size=256, verbose=0)
        total_classes = int(max(np.max(y_train), np.max(y_val))) + 1
        proto, missing = compute_prototypes(tr_proj, y_train, total_classes)
        logits = temperature * (va_proj @ proto.T)
        if missing:
            logits[:, missing] = -1e9
        valid = np.isin(y_val, np.unique(y_train))
        if not np.any(valid):
            raise ValueError('No validation classes are represented in training; adjust split')
        val_loss = float(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_val[valid], logits[valid], from_logits=True)).numpy())
        val_acc = float(np.mean(np.argmax(logits[valid], axis=1) == y_val[valid]))
        history['loss'].append(float(np.mean(losses)))
        history['accuracy'].append(float(np.mean(accuracies)))
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        print(f'epoch={epoch+1} loss={history["loss"][-1]:.4f} val_loss={val_loss:.4f} val_accuracy={val_acc:.4f}')
        if val_loss < best - 1e-6:
            best=val_loss;best_weights=encoder.get_weights();no_improve=0
        else:
            no_improve+=1
            if no_improve>=patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    if best_weights is not None:
        encoder.set_weights(best_weights)
    class History:
        pass
    result=History(); result.history=history
    return result
