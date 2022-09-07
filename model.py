import tensorflow as tf

def create_model(d:int,
                 s:int,
                 m:int,
                 upscaling:int,
                 input,
                 color_channels=1):
    # Input layer
    inputs = tf.keras.layers.Input((input.shape[0], input.shpae[1],color_channels))

    # Feature Extraction
    x = tf.keras.layers.Conv2D(filters = d,
                               kernel_size = 5,
                               )(inputs)
    x = tf.keras.layers.PReLU()(x)

    # Shrinking layer
    x = tf.keras.layers.Conv2D(filters = s,
                               kernel_size = 1)(x)
    x = tf.keras.layers.PReLU()(x)

    # Mapping layers
    for i in range(m):
        x = tf.keras.layers.Conv2D(filters = s,
                                   kernel_size = 3)(x)
        x = tf.keras.layers.PReLU()(x)

    # Expanding layer
    x = tf.keras.layers.Conv2D(filters = d,
                               kernel_size = 1)
    x = tf.keras.layers.PReLU()(x)

    # Deconvolution layer
    # Stride = upscaling factor
    outputs = tf.keras.layers.Conv2DTranspose(filters=color_channels,
                                              kernel_size = 9,
                                              strides=upscaling,
                                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.001, seed=None)
)

    model = tf.keras.models(inputs, outputs)

    model.compile(loss=tf.keras.losses.mse,
                  optimizer="adam")

    return model