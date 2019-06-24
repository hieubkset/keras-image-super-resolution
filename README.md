# Single Image Super Resolution, EDSR, SRGAN, SRFeat, RCAN, ESRGAN and ERCA (ours) benchmark comparison

This is a keras implementation of single super resolution algorithms: [EDSR](https://arxiv.org/abs/1707.02921), [SRGAN](https://arxiv.org/abs/1609.04802), [SRFeat](http://openaccess.thecvf.com/content_ECCV_2018/papers/Seong-Jin_Park_SRFeat_Single_Image_ECCV_2018_paper.pdf), [RCAN](https://arxiv.org/abs/1807.02758), [ESRGAN](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf) and ERCA (ours). This project is used for one of my courses, which aims to improve the performace of the baseline (SRFeat). 

To run this project you need to setup the environment, download the dataset, run script to process data, and then you can train and test the network models. I will show you step by step to run this project and i hope it is clear enough.

## Prerequiste
I tested my project in Corei7, 64G RAM, GPU Titan XP. Because it takes about several days for training, I recommend you using CPU/GPU strong enough and about 16 or 24G RAM.

## Environment
I recommend you using virtualenv to create a virtual environments. You can install virtualenv (which is itself a pip package) via
```
pip install virtualenv
```
Create a virtual environment called venv with python3, one runs
```
virtualenv -p python3 .env
```
Activate the virtual enviroment:
```
source .env/bin/activate
```
Install dependencies:
```
pip install -r requirements.txt
```
## Dataset
I use DIV2K dataset ([link](https://data.vision.ee.ethz.ch/cvl/DIV2K/)) which consists of 800 HR training images and 100 HR validation images. To expand the volume of training data, I applied data augmentation method as SRFeat. The author provides augmentation code. You can find it [here](https://github.com/HyeongseokSon1/SRFeat/tree/master/data_augmentation).

After applying agumentation, you shold see about 150 thousands of images for each folder (GT and LR_bicubic). 
## Training
To pretrain a generator, run the following command
```
python pretrain.py --arc=[edsr, srgan, srfeat, rcan, esrgan, erca] --train=/path/to/training/data --train-ext=[.png, .jpg] --valid=/path/to/validation/data --valid-ext=[.png, .jpg] [--resume=/path/to/checkpoint --init_epoch=0 --cuda=1
```
For example, to train a ERCA generator with DIV2K dataset:
```
python pretrain.py --arc=erca --train=data/train/DIV2K --train-ext=.png --valid=data/test/Set5 --valid-ext=.png --cuda=1
```
Data folders should consist of a HR folder and a LR folder, e.g: data/train/DIV2K/HR and data/train/DIV2K/LR.
To train a generator by using GAN, run the following command
```
python gantrain.py --arc=[edsr, srgan, srfeat, rcan, esrgan, erca] --train=/path/to/training/data --train-ext=[.png, .jpg] --g_init=/path/to/pretrain/model --cuda=1
```
For example:
```
python gantrain.py --arc=erca --train=data/train/DIV2K --train-ext=.png --g_init=exp/erca-06-24-21\:12/final_model.h5 --cuda=0
```
Please note that we only implement a gan algorithm that is same with SRFeat. 
## Generating Super-Resolution Image
To generate SR images from a trained model, you should able to run:
- For one image
```
python demo.py --arc=[edsr, srgan, srfeat, rcan, esrgan, erca] --lr_path=/path/to/one/image --save_dir=/path/to/save --model_path=/path/to/model --cuda=0
```
- For images in a folder
```
python demo.py --arc=[edsr, srgan, srfeat, rcan, esrgan, erca] --lr_dir=/path/to/folder --ext=[.png, .jpg] --save_dir=/path/to/save --model_path=/path/to/model --cuda=0
```
## Benchmark comparisions



I hope my instructions are clear enough for you. If you have any problem, you can contact me through hieubkset@gmail.com or use the issue tab. If you are insterested in this project, you are very welcome. Many Thanks.
