import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, PReLU, Add, Lambda
from tensorflow.keras.models import Model


def dense_block(input_tensor, filters, scale=0.2):
    x_1 = input_tensor

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = LeakyReLU(alpha=0.2)(x)
    x = x_2 = Add()([x_1, x])

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = x_3 = Add()([x_1, x_2, x])

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Add()([x_1, x_2, x_3, x])

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Add()([x_1, x_2, x_3, x])

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Lambda(lambda x: x * scale)(x)
    x = Add()([x_1, x])

    return x


def sub_pixel_conv2d(scale=2, **kwargs):
    return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)


def upsample(input_tensor, filters):
    x = Conv2D(filters=filters*4, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = sub_pixel_conv2d(scale=2)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    return x


def generator(filters=64, n_dense_block=16, n_sub_block=2):
    inputs = Input(shape=(None, None, 3))

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(inputs)
    x = x_1 = LeakyReLU(alpha=0.2)(x)

    for _ in range(n_dense_block):
        x = dense_block(x, filters=filters)

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = Lambda(lambda x: x * 0.2)(x)
    x = Add()([x_1, x])

    for _ in range(n_sub_block):
        x = upsample(x, filters)

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(x)

    return Model(inputs=inputs, outputs=x)
