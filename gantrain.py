import os
import time
import datetime
import argparse
import logging

import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error

import data
from model import get_generator
from model.discr import vgg_emb5, image_discriminator, feature_discriminator, discriminator_generator
from callbacks import make_lr_callback
from utils import save_params
from tensorboardX import SummaryWriter


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)


def content_loss(hr, sr, scale=1 / 12.75):
    sr_features = vgg_emb5(sr, scale)
    hr_features = vgg_emb5(hr, scale)
    return mean_squared_error(hr_features, sr_features)


def make_exp_folder(exp_dir, model_name):
    folder = os.path.join(exp_dir, model_name + '-gan-' + datetime.datetime.now().strftime("%m-%d-%H:%M"))
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def num_iter_per_epoch(num_data, batch_size):
    return int(num_data // batch_size + (num_data % batch_size > 0))


def prepare_model(**params):
    print("** Load initial generator at: " + params['g_init'])
    start = time.time()
    g = get_generator(params['arc'], is_train=False)
    g.load_weights(params['g_init'])
    print("Finish loading generator in %.2fs" % (time.time() - start))

    img_d = image_discriminator()
    img_d.compile(loss=binary_crossentropy, loss_weights=[params['per_loss_w']], optimizer=Adam(lr=params['lr_init']))
    img_lr_scheduler = make_lr_callback(params['lr_init'], params['lr_decay'], params['lr_decay_at_steps'])
    img_lr_scheduler.set_model(img_d)

    f_d = feature_discriminator()
    f_d.compile(loss='binary_crossentropy', loss_weights=[params['per_loss_w']], optimizer=Adam(lr=params['lr_init']))
    f_lr_scheduler = make_lr_callback(params['lr_init'], params['lr_decay'], params['lr_decay_at_steps'])
    f_lr_scheduler.set_model(f_d)

    d_g = discriminator_generator(g, img_d, f_d)
    d_g.compile(loss=[content_loss, 'binary_crossentropy', 'binary_crossentropy'],
                loss_weights=[1.0, params['per_loss_w'], params['per_loss_w']], optimizer=Adam(lr=params['lr_init']))
    d_g_lr_scheduler = make_lr_callback(params['lr_init'], params['lr_decay'], params['lr_decay_at_steps'])
    d_g_lr_scheduler.set_model(d_g)

    def on_epoch_begin(epoch):
        d_g_lr_scheduler.on_epoch_begin(epoch)
        img_lr_scheduler.on_epoch_begin(epoch)
        f_lr_scheduler.on_epoch_begin(epoch)

    def on_epoch_end(epoch):
        d_g_lr_scheduler.on_epoch_end(epoch)
        img_lr_scheduler.on_epoch_end(epoch)
        f_lr_scheduler.on_epoch_end(epoch)

    return g, img_d, f_d, d_g, on_epoch_begin, on_epoch_end


def train(**params):
    print("** Loading training images")
    start = time.time()
    lr_hr_ds, n_data = data.load_train_dataset(params['lr_dir'], params['hr_dir'], params['ext'], params['batch_size'])
    print("Finish loading images in %.2fs" % (time.time() - start))

    exp_folder = make_exp_folder(params['exp_dir'], params['arc'])
    save_params(exp_folder, **params)
    tensorboard = SummaryWriter(exp_folder)

    g, img_d, f_d, d_g, on_epoch_begin, on_epoch_end = prepare_model(**params)

    print("** Training")
    sess = tf.Session()
    n_iter = num_iter_per_epoch(n_data, params['batch_size'])
    for epoch in range(params['epochs']):
        on_epoch_begin(epoch)

        next_element = lr_hr_ds.get_next()

        for iteration in range(n_iter):
            step_time = time.time()

            lr, hr = sess.run(next_element)

            sr = g.predict(lr, steps=1)

            hr_img_d_loss = img_d.train_on_batch(hr, np.ones(sr.shape[0]))
            sr_img_d_loss = img_d.train_on_batch(sr, np.zeros(sr.shape[0]))

            hr_f_d_loss = f_d.train_on_batch(hr, np.ones(sr.shape[0]))
            sr_f_d_loss = f_d.train_on_batch(sr, np.zeros(sr.shape[0]))

            d_loss = hr_img_d_loss + sr_img_d_loss + hr_f_d_loss + sr_f_d_loss

            g_loss = d_g.train_on_batch(lr, [hr, np.ones(sr.shape[0]), np.ones(sr.shape[0])])

            print(
                "Epoch [%d/%d] [%d/%d] - %4.4fs,"
                "d_loss: %.8f (d1: %.8f, d2: %.8f, d1_vgg: %.8f, d2_vgg: %.8f), "
                "g_loss: %.8f (vgg: %.6f, adv: %.6f, adv2: %.6f)" % (
                    epoch + 1, params['epochs'], iteration, n_iter, time.time() - step_time,
                    d_loss, hr_img_d_loss, sr_img_d_loss, hr_f_d_loss, sr_f_d_loss,
                    g_loss[0], g_loss[1], g_loss[2], g_loss[3]))

            tensorboard.add_scalar('d/total_loss', d_loss, epoch * n_iter + iteration)
            tensorboard.add_scalar('d/loss1', hr_img_d_loss, epoch * n_iter + iteration)
            tensorboard.add_scalar('d/loss2', sr_img_d_loss, epoch * n_iter + iteration)
            tensorboard.add_scalar('d/vgg_loss1', hr_f_d_loss, epoch * n_iter + iteration)
            tensorboard.add_scalar('d/vgg_loss2', sr_f_d_loss, epoch * n_iter + iteration)
            tensorboard.add_scalar('g/total_loss', g_loss[0], epoch * n_iter + iteration)
            tensorboard.add_scalar('g/vgg_loss', g_loss[1], epoch * n_iter + iteration)
            tensorboard.add_scalar('g/gan_loss', g_loss[2], epoch * n_iter + iteration)
            tensorboard.add_scalar('g/gan_vgg_loss', g_loss[3], epoch * n_iter + iteration)

            if iteration % 1000 == 0:
                g.save_weights(os.path.join(exp_folder, "gan-cp-%05d-%02d.h5" % (iteration + 1, epoch + 1)))

        on_epoch_end(epoch)

        g.save_weights(os.path.join(exp_folder, "gan-cp-%02d.h5" % (epoch + 1)))

    K.clear_session()


def main():
    parser = argparse.ArgumentParser(description='Single Image Super-Resolution')
    parser.add_argument('--arc', type=str, required=True, help='Model type?')
    parser.add_argument('--g_init', type=str, required=True, help='Path to a pre-trained generator')
    parser.add_argument('--train', type=str, required=True, help='Path to training data')
    parser.add_argument('--train-ext', type=str, required=True, help='Extension of training images')
    parser.add_argument('--cuda', type=str, help='a list of gpus')
    args = parser.parse_args()

    if args.cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
        n_gpus = len(args.cuda.split(','))
    else:
        print('Training without gpu. It is recommended using at least one gpu.')
        n_gpus = 0

    params = {
        'arc': args.arc,
        'g_init': args.g_init,
        'n_gpus': n_gpus,
        #
        'epochs': 5,
        'lr_init': 1e-4,
        'lr_decay': 0.1,
        'lr_decay_at_steps': [3, 5],
        #
        'per_loss_w': 1e-3,
        #
        'patch_size_lr': 74,
        'path_size_hr': 296,
        #
        'hr_dir': os.path.join(args.train, 'HR'),
        'lr_dir': os.path.join(args.train, 'LR'),
        'ext': '.png',
        'batch_size': 8,
        #
        'exp_dir': './exp/',
    }

    train(**params)


if __name__ == '__main__':
    main()

# python gantrain.py --arc=erca --train=../SRFeat/data/train/DIV2K --train-ext=.png --g_init=exp/erca-06-24-21\:12/final_model.h5 --cuda=0
