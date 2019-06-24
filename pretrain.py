import os
import time
import datetime
import argparse

from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error

import data
from model import get_generator
from metrics import psnr
from utils import save_params, num_iter_per_epoch
from callbacks import make_tb_callback, make_lr_callback, make_cp_callback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_model(model, path):
    if path is not None:
        print("** Load model at: " + path)
        model.load_weights(path)
    return model


def make_gpu_model(model, n_gpus):
    if n_gpus > 1:
        gpu_model = multi_gpu_model(model, gpus=n_gpus)
    else:
        gpu_model = model
    return gpu_model


def make_exp_folder(exp_dir, model_name):
    folder = os.path.join(exp_dir, model_name + '-' + datetime.datetime.now().strftime("%m-%d-%H:%M"))
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def adaptive_batch_size(n_gpus):
    if n_gpus < 3:
        batch_size = 16
    else:
        batch_size = 32
    return batch_size


def prepare_model(**params):
    model_arc = params['arc']
    model = get_generator(model_arc)
    
    if model_arc == 'srfeat' or model_arc == 'srgan':
        loss = mean_squared_error
    else:
        loss = mean_absolute_error

    model = load_model(model, params['resume'])
    gpu_model = make_gpu_model(model, params['n_gpus'])
    optimizer = Adam(lr=params['lr_init'])
    gpu_model.compile(optimizer=optimizer, loss=loss, metrics=[psnr])

    return model, gpu_model


def train(**params):
    print("** Loading training images")
    start = time.time()
    lr_hr_ds, n_data = data.load_train_dataset(params['lr_dir'], params['hr_dir'], params['ext'], params['batch_size'])
    val_lr_hr_ds, n_val_data = data.load_test_dataset(params['val_lr_dir'], params['val_hr_dir'], params['val_ext'],
                                                      params['val_batch_size'])
    print("Finish loading images in %.2fs" % (time.time() - start))

    one_gpu_model, gpu_model = prepare_model(**params)

    exp_folder = make_exp_folder(params['exp_dir'], params['arc'])
    save_params(exp_folder, **params)
    tb_callback = make_tb_callback(exp_folder)
    lr_callback = make_lr_callback(params['lr_init'], params['lr_decay'], params['lr_decay_at_steps'])
    cp_callback = make_cp_callback(exp_folder, one_gpu_model)

    gpu_model.fit(lr_hr_ds, epochs=params['epochs'],
                  steps_per_epoch=num_iter_per_epoch(n_data, params['batch_size']),
                  callbacks=[tb_callback, cp_callback, lr_callback],
                  initial_epoch=params['init_epoch'],
                  validation_data=val_lr_hr_ds,
                  validation_steps=n_val_data)

    one_gpu_model.save_weights(os.path.join(exp_folder, 'final_model.h5'))

    K.clear_session()


def main():
    parser = argparse.ArgumentParser(description='Single Image Super-Resolution')
    parser.add_argument('--arc', type=str, required=True, help='Model type?')
    parser.add_argument('--train', type=str, required=True, help='Path to training data')
    parser.add_argument('--train-ext', type=str, required=True, help='Extension of training images')
    parser.add_argument('--valid', type=str, required=True, help='Path to validation data')
    parser.add_argument('--valid-ext', type=str, required=True, help='Extension of validation images')
    parser.add_argument('--resume', type=str, default=None, help='Path to a checkpoint')
    parser.add_argument('--init_epoch', type=int, default=0, help="Initial epoch")
    parser.add_argument('--cuda', type=str, default=None, help='a list of gpus')
    args = parser.parse_args()

    if args.cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
        n_gpus = len(args.cuda.split(','))
        batch_size = adaptive_batch_size(n_gpus)
    else:
        print('Training without gpu. It is recommended using at least one gpu.')
        n_gpus = 0
        batch_size = 8

    params = {
        'arc': args.arc,
        'resume': args.resume,
        'init_epoch': args.init_epoch,
        'n_gpus': n_gpus,
        #
        'epochs': 20,
        'lr_init': 1e-4,
        'lr_decay': 0.5,
        'lr_decay_at_steps': [10, 15],
        #
        'patch_size_lr': 74,
        'path_size_hr': 296,
        #
        'hr_dir': os.path.join(args.train, 'HR'),
        'lr_dir': os.path.join(args.train, 'LR'),
        'ext': args.train_ext,
        'batch_size': batch_size,
        #
        'val_hr_dir': os.path.join(args.valid, 'HR'),
        'val_lr_dir': os.path.join(args.valid, 'LR'),
        'val_ext': args.valid_ext,
        'val_batch_size': 1,
        #
        'exp_dir': './exp/',
    }

    train(**params)


if __name__ == '__main__':
    main()

# python pretrain.py --arc=erca --train=../SRFeat/data/train/DIV2K --train-ext=.png --valid=../SRFeat/data/test/Set5 --valid-ext=.png --resume=exp/erca-06-24-21\:05/cp-0014.h5 --init_epoch=14 --cuda=1
