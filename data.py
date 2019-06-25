import random
import pathlib
import tensorflow as tf


def preprocess_image(image, ext):
    """
    Normalize image to [-1, 1]
    """
    assert ext in ['.png', '.jpg', '.jpeg', '.JPEG']
    if ext == '.png':
        image = tf.image.decode_png(image, channels=3)
    else:
        image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image -= 0.5
    image /= 0.5

    return image


def load_and_preprocess_image(image_path, ext):
    image = tf.read_file(image_path)
    return preprocess_image(image, ext)


def get_sorted_image_path(path, ext):
    ext_regex = "*" + ext
    data_root = pathlib.Path(path)
    image_paths = list(data_root.glob(ext_regex))
    image_paths = sorted([str(path) for path in image_paths])
    return image_paths


def get_dataset(lr_path, hr_path, ext):
    lr_sorted_paths = get_sorted_image_path(lr_path, ext)
    hr_sorted_paths = get_sorted_image_path(hr_path, ext)

    lr_hr_sorted_paths = list(zip(lr_sorted_paths[:], hr_sorted_paths[:]))
    random.shuffle(lr_hr_sorted_paths)
    lr_sorted_paths, hr_sorted_paths = zip(*lr_hr_sorted_paths)

    ds = tf.data.Dataset.from_tensor_slices((list(lr_sorted_paths), list(hr_sorted_paths)))

    def load_and_preprocess_lr_hr_images(lr_path, hr_path, ext=ext):
        return load_and_preprocess_image(lr_path, ext), load_and_preprocess_image(hr_path, ext)

    lr_hr_ds = ds.map(load_and_preprocess_lr_hr_images, num_parallel_calls=8)
    return lr_hr_ds, len(lr_sorted_paths)


def load_train_dataset(lr_path, hr_path, ext, batch_size):
    lr_hr_ds, n_data = get_dataset(lr_path, hr_path, ext)
    lr_hr_ds = lr_hr_ds.batch(batch_size)
    lr_hr_ds = lr_hr_ds.repeat()
    lr_hr_ds = lr_hr_ds.make_one_shot_iterator()
    return lr_hr_ds, n_data


def load_test_dataset(lr_path, hr_path, ext, batch_size):
    val_lr_hr_ds, val_n_data = get_dataset(lr_path, hr_path, ext)
    val_lr_hr_ds = val_lr_hr_ds.batch(batch_size)
    val_lr_hr_ds = val_lr_hr_ds.repeat()
    return val_lr_hr_ds, val_n_data
