import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Add, Lambda, GlobalAveragePooling2D, Multiply, Dense, Reshape
from tensorflow.keras.models import Model


def ca(input_tensor, filters, reduce=16):
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    a = GlobalAveragePooling2D()(x)
    a = Reshape((1, 1, filters))(a)
    a = Dense(filters/reduce,  activation='relu', kernel_initializer='he_normal', use_bias=False)(a)
    a = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(a)

    x = Multiply()([x, a])

    return x


def identity_block(input_tensor, filters):
    x = ca(input_tensor, filters)
    x = Add()([x, input_tensor])
    return x


def sub_pixel_conv2d(scale=2, **kwargs):
    return Lambda(lambda x: tf.depth_to_space(x, scale), **kwargs)


def upsample(input_tensor, filters):
    x = Conv2D(filters=filters*4, kernel_size=3, strides=1, padding='same')(input_tensor)
    x = sub_pixel_conv2d(scale=2)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def generator(filters=128, n_id_block=16, n_sub_block=2):
    inputs = Input(shape=(None, None, 3))

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(inputs)

    x_1 = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)

    for _ in range(n_id_block):
        x = identity_block(x, filters=filters)

    x = Add()([x_1, x])

    for _ in range(n_sub_block):
        x = upsample(x, filters)

    x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(x)

    return Model(inputs=inputs, outputs=x)
