import tensorflow as tf

def create_model(d:int,
                 s:int,
                 m:int,
                 rescaling:int,
                 input,
                 color_channels=3):
    # Input layer
    inputs = tf.keras.layers.Input((input.shape[0], input.shpae[1],color_channels))

    # Feature Extraction
    x = tf.keras.layers.Conv2D(filters = d,
                               kernel_size = 5,
                               padding="same",
                               kernel_initializer=tf.keras.initializers.HeNormal)(inputs)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    # Shrinking layer
    x = tf.keras.layers.Conv2D(filters = s,
                               kernel_size = 1,
                               padding="same",
                               kernel_initializer=tf.keras.initializers.HeNormal)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    # Mapping layers
    for i in range(m):
        x = tf.keras.layers.Conv2D(filters = s,
                                   kernel_size = 3,
                                   padding="same",
                                   kernel_initializer=tf.keras.initializers.HeNormal)(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    # Expanding layer
    x = tf.keras.layers.Conv2D(filters = d,
                               kernel_size = 1,
                               padding="same",
                               kernel_initializer=tf.keras.initializers.HeNormal)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    # Deconvolution layer
    # Stride = upscaling factor
    outputs = tf.keras.layers.Conv2DTranspose(filters=color_channels,
                                              kernel_size = 9,
                                              strides=rescaling,
                                              padding="same",
                                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None)(x)
)

    model = tf.keras.models(inputs, outputs)

    model.compile(loss=tf.keras.losses.mse,
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))

    return model


def get_callbacks():

    return [tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5, patience=10),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=25,restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(monitor="loss", save_best_only=True, save_weights_only=True)]