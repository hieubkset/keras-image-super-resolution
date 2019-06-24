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
