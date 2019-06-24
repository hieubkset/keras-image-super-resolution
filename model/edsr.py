import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, Lambda
from tensorflow.keras.models import Model


def res_block(input_tensor, filters, scale=0.1):
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    if scale:
        x = Lambda(lambda t: t * scale)(x)
    x = Add()([x, input_tensor])

    return x


def sub_pixel_conv2d(scale=2, **kwargs):
    return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)


def upsample(input_tensor, filters):
    x = Conv2D(filters=filters * 4, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = sub_pixel_conv2d(scale=2)(x)
    x = Activation('relu')(x)
    return x


def generator(filters=128, n_id_block=16, n_sub_block=2):
    inputs = Input(shape=(None, None, 3))

    x = x_1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(inputs)

    for _ in range(n_id_block):
        x = res_block(x, filters=filters)

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    x = Add()([x_1, x])

    for _ in range(n_sub_block):
        x = upsample(x, filters)
    x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(x)

    return Model(inputs=inputs, outputs=x)
