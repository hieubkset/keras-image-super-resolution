import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, Lambda, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

LR_SIZE = 74
HR_SIZE = 296


def discriminator_block(inputs, num_filters, strides=1, bn=True):
    x = Conv2D(filters=num_filters, kernel_size=3, strides=strides, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    if bn:
        x = BatchNormalization()(x)
    return x


def discriminator(input_shape, d_name, n_blocks=4, num_filters=64):
    inputs = Input(shape=input_shape)

    if d_name == 'vgg_d':
        x = Lambda(vgg_emb5)(inputs)
    else:
        x = inputs

    for i in range(n_blocks):
        if i == 0:
            x = discriminator_block(x, num_filters * (1 << i), strides=1, bn=False)
        else:
            x = discriminator_block(x, num_filters * (1 << i), strides=1)
        x = discriminator_block(x, num_filters * (1 << i), strides=2)

    x = Flatten()(x)

    x = Dense(units=1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(units=1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=x)


def image_discriminator():
    return discriminator(input_shape=(HR_SIZE, HR_SIZE, 3), d_name='d')


def feature_discriminator():
    return discriminator(input_shape=(HR_SIZE, HR_SIZE, 3), d_name='vgg_d')


def discriminator_generator(generator, image_discriminator, feature_discriminator):
    image_discriminator.trainable = False
    feature_discriminator.trainable = False

    x_in = Input(shape=(LR_SIZE, LR_SIZE, 3))

    x_1 = generator(x_in)

    x_2 = image_discriminator(x_1)

    x_3 = feature_discriminator(x_1)

    return Model(inputs=x_in, outputs=[x_1, x_2, x_3])


def _vgg_input_rescale(inputs):
    x = (inputs + 1)/2
    x = tf.maximum(0.0, tf.minimum(x, 1.0))
    x = x * 255
    return x


def vgg_emb5(inputs, scale=1 / 12.75):
    return scale * _vgg(20)(preprocess_input(_vgg_input_rescale(inputs)))


def _vgg(output_layer):
    vgg = VGG19(weights='imagenet', input_shape=(HR_SIZE, HR_SIZE, 3), include_top=False)
    mdl = Model(vgg.input, vgg.layers[output_layer].output)
    mdl.trainable = False
    return mdl
