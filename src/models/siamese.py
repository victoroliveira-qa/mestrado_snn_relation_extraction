import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="SNNResearch")
class L2Normalize(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)


@tf.keras.utils.register_keras_serializable(package="SNNResearch")
class AbsoluteDifference(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs[0]-inputs[1])


def build_model(input_dim, cfg, learning_rate):
    reg = tf.keras.regularizers.l2(float(cfg["l2_regularization"]))
    drop = float(cfg["dropout"])
    sizes = cfg["projection_dims"]
    heads = cfg["head_dims"]
    inp = tf.keras.Input(shape=(input_dim,),name="branch_input")
    z = inp
    for j,width in enumerate(sizes,1):
        z = tf.keras.layers.Dense(width,kernel_regularizer=reg,name=f"proj_fc{j}")(z)
        z = tf.keras.layers.BatchNormalization(name=f"proj_bn{j}")(z)
        z = tf.keras.layers.ReLU(name=f"proj_relu{j}")(z)
        z = tf.keras.layers.Dropout(drop,name=f"proj_drop{j}")(z)
    encoder = tf.keras.Model(inp,L2Normalize(name="proj_l2_norm")(z),name="shared_projection_branch")
    left=tf.keras.Input(shape=(input_dim,),name="left_embedding")
    right=tf.keras.Input(shape=(input_dim,),name="right_embedding")
    a,b=encoder(left),encoder(right)
    diff=AbsoluteDifference(name="abs_diff")([a,b])
    mult=tf.keras.layers.Multiply(name="elementwise_mul")([a,b])
    fusion=cfg.get("fusion","v3_ordered")
    if fusion=="v3_ordered":
        features=[a,b,diff,mult]
    elif fusion=="symmetric":
        features=[diff,mult]
    else:
        raise ValueError("fusion must be v3_ordered or symmetric")
    z=tf.keras.layers.Concatenate(name="fusion_concat")(features)
    for j,width in enumerate(heads,1):
        z=tf.keras.layers.Dense(width,kernel_regularizer=reg,name=f"head_fc{j}")(z)
        z=tf.keras.layers.BatchNormalization(name=f"head_bn{j}")(z)
        z=tf.keras.layers.ReLU(name=f"head_relu{j}")(z)
        z=tf.keras.layers.Dropout(drop,name=f"head_drop{j}")(z)
    out=tf.keras.layers.Dense(1,activation="sigmoid",name="pair_output")(z)
    model=tf.keras.Model([left,right],out,name="qwen_siamese_mlp_classifier")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),loss="binary_crossentropy",metrics=["accuracy",tf.keras.metrics.AUC(name="auc")])
    return model,encoder
