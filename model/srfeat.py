import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Add, Lambda
from tensorflow.keras.models import Model


def identity_block(input_tensor, filters, is_train, use_bn, use_bias=False):
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', use_bias=use_bias)(input_tensor)
    if use_bn:
        x = BatchNormalization(trainable=is_train)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', use_bias=use_bias)(x)
    if use_bn:
        x = BatchNormalization(trainable=is_train)(x)

    x = Add()([x, input_tensor])

    long_skip_con = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same', use_bias=use_bias)(x)

    return x, long_skip_con


def sub_pixel_conv2d(scale=2, **kwargs):
    return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)


def upsample(input_tensor, filters):
    x = Conv2D(filters=filters*4, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = sub_pixel_conv2d(scale=2)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def generator(filters=128, n_id_block=16, n_sub_block=2, use_bn=True, is_train=True):
    inputs = Input(shape=(None, None, 3))

    x = Conv2D(filters=filters, kernel_size=9, strides=1, padding='same')(inputs)

    long_skip_cons = []
    for _ in range(n_id_block):
        x, long_skip_con = identity_block(x, filters=filters, is_train=is_train, use_bn=use_bn)
        long_skip_cons.append(long_skip_con)

    long_skip_cons[-1] = x
    x = Add()(long_skip_cons)

    for _ in range(n_sub_block):
        x = upsample(x, filters)

    x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(x)

    return Model(inputs=inputs, outputs=x)
