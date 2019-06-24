import tensorflow as tf


def rgb_to_y(image):
    image = tf.image.rgb_to_yuv(image)
    image = (image * (235 - 16) + 16) / 255.0
    return image[:, :, :, 0]


def crop(image):
    margin = 4
    image = image[:, margin:-margin, margin:-margin]
    return tf.expand_dims(image, -1)


def un_normalize(hr, sr):
    hr = hr * 0.5 + 0.5
    sr = tf.clip_by_value(sr, -1, 1)
    sr = sr * 0.5 + 0.5
    return hr, sr


def psnr(hr, sr):
    hr, sr = un_normalize(hr, sr)
    hr = rgb_to_y(hr)
    sr = rgb_to_y(sr)
    hr = crop(hr)
    sr = crop(sr)
    return tf.image.psnr(hr, sr, max_val=1.0)


def ssim(hr, sr):
    hr, sr = un_normalize(hr, sr)
    hr = rgb_to_y(hr)
    sr = rgb_to_y(sr)
    hr = crop(hr)
    sr = crop(sr)
    return tf.image.ssim(hr, sr, max_val=1.0)
