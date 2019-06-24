import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, PReLU, BatchNormalization, Add, Lambda
from tensorflow.keras.models import Model


def normalization_01(**kwargs):
    return Lambda(lambda x: (x + 1) / 2.0, **kwargs)


def denormalization_11(**kwargs):
    return Lambda(lambda x: (x - 0.5) / 0.5, **kwargs)


def identity_block(input_tensor, filters, is_train, use_bn):
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(input_tensor)
    if use_bn:
        x = BatchNormalization(trainable=is_train)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    if use_bn:
        x = BatchNormalization(trainable=is_train)(x)

    x = Add()([x, input_tensor])

    return x


def sub_pixel_conv2d(scale=2, **kwargs):
    return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)


def upsample(input_tensor, filters):
    x = Conv2D(filters=filters * 4, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = sub_pixel_conv2d(scale=2)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    return x


def generator(filters=64, n_id_block=16, n_sub_block=2, use_bn=True, is_train=True):
    inputs = Input(shape=(None, None, 3))
    x = normalization_01()(inputs)

    x = Conv2D(filters=filters, kernel_size=9, strides=1, padding='same')(x)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)

    for _ in range(n_id_block):
        x = identity_block(x, filters=filters, is_train=is_train, use_bn=use_bn)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    if use_bn:
        x = BatchNormalization(trainable=is_train)(x)
    x = Add()([x_1, x])

    for _ in range(n_sub_block):
        x = upsample(x, filters)

    x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(x)

    x = denormalization_11()(x)

    return Model(inputs=inputs, outputs=x)
