import tensorflow as tf
import numpy as np 

def channel_padding(x):
    """
    zero padding in an axis of channel 
    """

    return tf.keras.backend.concatenate([x, tf.zeros_like(x)], axis=-1)


def singleBlazeBlock(x, filters=24, kernel_size=5, strides=1, padding='same'):

    # depth-wise separable convolution
    x_0 = tf.keras.layers.SeparableConv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False)(x)

    x_1 = tf.keras.layers.BatchNormalization()(x_0)

    # Residual connection

    if strides == 2:
        input_channels = x.shape[-1]
        output_channels = x_1.shape[-1]

        x_ = tf.keras.layers.MaxPooling2D()(x)

        if output_channels - input_channels != 0:

            # channel padding
            x_ = tf.keras.layers.Lambda(channel_padding)(x_)

        out = tf.keras.layers.Add()([x_1, x_])
        return tf.keras.layers.Activation("relu")(out)

    out = tf.keras.layers.Add()([x_1, x])
    return tf.keras.layers.Activation("relu")(out)


def doubleBlazeBlock(x, filters_1=24, filters_2=96,
                     kernel_size=5, strides=1, padding='same'):

    # depth-wise separable convolution, project
    x_0 = tf.keras.layers.SeparableConv2D(
        filters=filters_1,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False)(x)

    x_1 = tf.keras.layers.BatchNormalization()(x_0)

    x_2 = tf.keras.layers.Activation("relu")(x_1)

    # depth-wise separable convolution, expand
    x_3 = tf.keras.layers.SeparableConv2D(
        filters=filters_2,
        kernel_size=kernel_size,
        strides=1,
        padding=padding,
        use_bias=False)(x_2)

    x_4 = tf.keras.layers.BatchNormalization()(x_3)

    # Residual connection

    if strides == 2:
        input_channels = x.shape[-1]
        output_channels = x_4.shape[-1]

        x_ = tf.keras.layers.MaxPooling2D()(x)

        if output_channels - input_channels != 0:

            # channel padding
            x_ = tf.keras.layers.Lambda(channel_padding)(x_)

        out = tf.keras.layers.Add()([x_4, x_])
        return tf.keras.layers.Activation("relu")(out)

    out = tf.keras.layers.Add()([x_4, x])
    return tf.keras.layers.Activation("relu")(out)


def network(input_shape):

    inputs = tf.keras.layers.Input(shape=input_shape)

    x_0 = tf.keras.layers.Conv2D(
        filters=24, kernel_size=5, strides=2, padding='same')(inputs)
    x_0 = tf.keras.layers.BatchNormalization()(x_0)
    x_0 = tf.keras.layers.Activation("relu")(x_0)

    # single BlazeBlock phase
    x_1 = singleBlazeBlock(x_0)
    x_2 = singleBlazeBlock(x_1)
    x_3 = singleBlazeBlock(x_2, strides=2, filters=48)
    x_4 = singleBlazeBlock(x_3, filters=48)
    x_5 = singleBlazeBlock(x_4, filters=48)

    # double BlazeBlock phase

    x_6 = doubleBlazeBlock(x_5, strides=2)
    x_7 = doubleBlazeBlock(x_6)
    x_8 = doubleBlazeBlock(x_7)
    x_9 = doubleBlazeBlock(x_8, strides=2)
    x10 = doubleBlazeBlock(x_9)
    x11 = doubleBlazeBlock(x10)

    model = tf.keras.models.Model(inputs=inputs, outputs=[x_8, x11])
    return model


if __name__ == "__main__":
    pass

