import os
import glob
import pickle


def get_image_paths(images_dir, ext):
    return glob.glob(os.path.join(images_dir, "*" + ext))


def get_filename(image_path):
    filename_w_ext = os.path.basename(image_path)
    filename = os.path.splitext(filename_w_ext)[0]
    return filename


def get_file_ext(image_path):
    filename_w_ext = os.path.basename(image_path)
    ext = os.path.splitext(filename_w_ext)[1]
    return ext


def save_params(exp_folder, **params):
    file_name = 'params.pkl'
    file_path = os.path.join(exp_folder, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)


def load_params(exp_folder):
    file_name = 'params.pkl'
    file_path = os.path.join(exp_folder, file_name)
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def num_iter_per_epoch(num_data, batch_size):
    return int(num_data // batch_size + (num_data % batch_size > 0))
