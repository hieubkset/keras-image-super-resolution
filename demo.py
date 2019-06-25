import os
import argparse

import tensorflow as tf

import data
from model import get_generator
import utils


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def save_image(image, save_dir, file_name, ext):
    image = (image * 127.5) + 127.5
    image = tf.cast(image, tf.uint8)
    image = tf.squeeze(image, axis=0)

    if ext == ".png":
        image = tf.image.encode_png(image)
    else:
        image = tf.image.encode_jpeg(image, quality=100, format='rgb')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    full_sr_path = os.path.join(save_dir, file_name + ext)
    writer = tf.write_file(full_sr_path, image)
    sess.run(writer)
    print("Save a sr image at {}".format(full_sr_path))


def get_image(image_path, ext):
    image = data.load_and_preprocess_image(image_path, ext)
    image = tf.expand_dims(image, axis=0)
    return image


def sr_from_path(model, lr_path, save_dir):
    ext = utils.get_file_ext(lr_path)
    lr_image = get_image(lr_path, ext)

    sr_image = model.predict(lr_image, steps=1)
    sr_image = sr_image.clip(-1, 1)

    lr_filename = utils.get_filename(lr_path)
    sr_filename = lr_filename
    save_image(sr_image, save_dir, sr_filename, ext)


def sr_from_folder(model, lr_dir, save_dir, ext):
    if lr_dir is not None:
        if not os.path.exists(lr_dir):
            raise Exception('Not found folder: ' + lr_dir)
        lr_paths = utils.get_image_paths(lr_dir, ext)
        for lr_path in lr_paths:
            sr_from_path(model, lr_path, save_dir)


def main():
    parser = argparse.ArgumentParser(description='Generate SR images')
    parser.add_argument('--arc', required=True, type=str, help='Model architecture')
    parser.add_argument('--model_path', required=True, type=str, help='Path to a model')
    parser.add_argument('--lr_dir', type=str, default=None, help='Path to lr images')
    parser.add_argument('--lr_path', type=str, default=None, help='Path to a lr image')
    parser.add_argument('--ext', type=str, help='Image extension')
    parser.add_argument('--default', action='store_true', help='Path to lr images')
    parser.add_argument('--save_dir', type=str, help='folder to save SR images')
    parser.add_argument('--cuda', type=str, default=None, help='a list of gpus')
    args = parser.parse_args()

    if args.cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    global sess
    sess = tf.Session()
    model = get_generator(args.arc, is_train=False)
    print("** Loading model at: " + args.model_path)
    model.load_weights(args.model_path)

    if args.default:
        lr_dirs = [os.path.join("./data/test/", dataset, "LR") for dataset in ["Set5", "Set14", "BSDS100"]]
        save_dirs = [os.path.join("./output/", args.arc, dataset) for dataset in ["Set5", "Set14", "BSDS100"]]
        for lr_dir, save_dir in zip(lr_dirs, save_dirs):
            sr_from_folder(model, lr_dir, save_dir, ".png")
    else:
        sr_from_folder(model, args.lr_dir, args.save_dir, args.ext)
        if args.lr_path is not None:
            sr_from_path(model, args.lr_path, args.save_dir)


if __name__ == '__main__':
    main()

# python demo.py --default --arc=erca --model_path=exp/erca-06-24-21\:12/final_model.h5 --cuda=0
# python demo.py --arc=erca --lr_path=../SRFeat/data/test/Set5/LR/head.png --save_dir=./output/Set5 --model_path=exp/erca-06-24-21\:12/final_model.h5 --cuda=0
# python demo.py --arc=erca --lr_dir=../SRFeat/data/test/Set5/LR --ext=.png --save_dir=./output/Set5 --model_path=exp/erca-06-24-21\:12/final_model.h5 --cuda=0