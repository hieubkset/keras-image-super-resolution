# Single Image Super Resolution, EDSR, SRGAN, SRFeat, RCAN, ESRGAN and ERCA (ours) benchmark comparison

This is a keras implementation of single super resolution algorithms: [EDSR](https://arxiv.org/abs/1707.02921), [SRGAN](https://arxiv.org/abs/1609.04802), [SRFeat](http://openaccess.thecvf.com/content_ECCV_2018/papers/Seong-Jin_Park_SRFeat_Single_Image_ECCV_2018_paper.pdf), [RCAN](https://arxiv.org/abs/1807.02758), [ESRGAN](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.pdf) and [ERCA](https://drive.google.com/open?id=1GFEMT8rCR7SovhudMWFP_lvP_DrtHoTP) (ours). This project aims to improve the performace of the baseline (SRFeat). 

To run this project you need to setup the environment, download the dataset, run script to process data, and then you can train and test the network models. I will show you step by step to run this project and i hope it is clear enough.

## Prerequiste
I tested my project in Corei7, 64G RAM, GPU Titan XP. Because it takes about several days for training, I recommend you using CPU/GPU strong enough and about 12G RAM.

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

Actually, DIV2K dataset only contains high resolution images (HR images) and does not contains low resolution images (LR images).
So to run the code, you have to generate LR images first. You can do it by using matlab scripts (https://github.com/hieubkset/Keras-Image-Super-Resolution/tree/master/data_preprocess).

For training LR images, there are two scripts:
- aug_data_div2k.m: generate LR images by using bicubic interpolation with scale 4.
- aug_data_div2k_half.m: generate LR images by using bicubic interpolation with scale 2.

If you run both scripts, you shold see about 150 thousands of images for each folder (GT and LR_bicubic).

For testing LR images, using the script testset_bicubic_downsample.m

These scripts will search all HR images from a HR folder, and then generate LR images to a LR folder.
So you need to modify first lines of these scripts to your HR and LR folder.
 
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
- To generate SR images using our gan-trained model, run the following command:
```
python demo.py --arc=gan --lr_path=/path/to/one/image --save_dir=/path/to/save --model_path=/path/to/model --cuda=0
```
## Benchmark comparisions

<table>
  <tr>
    <th colspan="2" rowspan="2">Model</th>
    <th colspan="3">PSNR</th>
    <th colspan="3">SSIM</th>
    <th rowspan="2">Time per iteration<br>(s)</th>
    <th rowspan="2">Time per epoch</th>
  </tr>
  <tr>
    <td>Set5</td>
    <td>Set14</td>
    <td>BSDS100</td>
    <td>Set5</td>
    <td>Set14</td>
    <td>BSDS100</td>
  </tr>
  <tr>
    <td colspan="2">EDSR-10</td>
    <td>32.01</td>
    <td>28.56</td>
    <td>27.54</td>
    <td>0.8918</td>
    <td>0.7819</td>
    <td>0.7357</td>
    <td>0.3962</td>
    <td>1h 3min</td>
  </tr>
  <tr>
    <td colspan="2">SRGAN-10</td>
    <td>31.75</td>
    <td>28.39</td>
    <td>27.44</td>
    <td>0.8864</td>
    <td>0.7761</td>
    <td>0.7308</td>
    <td>0.3133</td>
    <td>50 min</td>
  </tr>
  <tr>
    <td colspan="2">ESRGAN-10</td>
    <td>31.90</td>
    <td>28.47</td>
    <td>27.49</td>
    <td>0.8898</td>
    <td>0.7789</td>
    <td>0.7340</td>
    <td>0.5265</td>
    <td>1h 24min</td>
  </tr>
  <tr>
    <td colspan="2">RCAN-10</td>
    <td>32.12</td>
    <td>28.65</td>
    <td>27.60</td>
    <td>0.8934</td>
    <td>0.7840</td>
    <td>0.7379</td>
    <td>1.2986</td>
    <td>3h 27min</td>
  </tr>
  <tr>
    <td colspan="2">SRFeat-10</td>
    <td>31.45</td>
    <td>28.17</td>
    <td>27.39</td>
    <td>0.8813</td>
    <td>0.7699</td>
    <td>0.7245</td>
    <td>0.5705</td>
    <td>1h 31min</td>
  </tr>
  <tr>
    <td colspan="2">Ours-10</td>
    <td>32.14</td>
    <td>28.60</td>
    <td>27.58</td>
    <td>0.8926</td>
    <td>0.7823</td>
    <td>0.7362</td>
    <td>0.5333</td>
    <td>1h 25min</td>
  </tr>
  <tr>
    <td colspan="2">SRFeat-20</td>
    <td>31.74</td>
    <td>28.34</td>
    <td>27.39</td>
    <td>0.8859</td>
    <td>0.7748</td>
    <td>0.7298</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="2">Ours-20</td>
    <td>32.21</td>
    <td>28.66</td>
    <td>27.60</td>
    <td>0.8936</td>
    <td>0.7836</td>
    <td>0.7370</td>
    <td></td>
    <td></td>
  </tr>
</table>

Model-10: after training 10 epochs.
Model-20: after training 20 epochs.

We run all with batch size of 16 and about 9600 iteration per epoch. Running time is reported using a GPU Titan XP 16G. We also find that training on a GPU Titan X 16G is much slower, for example, RCAN takes about 2.5s per iteration.

EDSR: in the paper, the author reported results of a model with 32 residual blocks and 256 features. The version here is one with 16 residual blocks and 128 filters.


## Learning Curves
![](/figs/learning_curves.png)

I hope my instructions are clear enough for you. If you have any problem, you can contact me through hieubkset@gmail.com or use the issue tab. If you are insterested in this project, you are very welcome. Many Thanks.
