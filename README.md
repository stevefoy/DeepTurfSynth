# Turfgrass analysis in PyTorch
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)


<img src="./sample_images/front.png" name="RGB">

PyTorch test implementation for our [paper](https://arxiv.org/).  If you find this code useful in your research, please consider citing

<!-- TOC -->

- [Turfgrass analysis in PyTorch]
  - [Requirements](#requirements)
  - [Main Features](#main-features)
    - [Models](#models)
    - [Datasets](#datasets)
  - [Code structure](#code-structure)
  - [Config file format](#config-file-format)
  - [Acknowledgement](#acknowledgement)
  - [Cite](#cite)

<!-- /TOC -->



## Requirements
Conda was used, config as follows:

```bash
pip install -r requirements.txt
```

or for a local installation

```bash
pip install --user -r requirements.txt
```
###manual install 
conda create enviroment
conda install -c conda-forge pytorch-gpu
Note: Add opencv via torchvision, I had issues on mid training crashes on dataloader code
conda install torchvision

## Main Features

- This code base is build on the shoulders of giants, please see  Acknowledgement below
- A `json` config file with a lot of possibilities for parameter tuning,
- Supports various models, losses, Lr schedulers, data augmentations and datasets,

**So, what's available ?**

### Models in code used
- [x] (**Deeplab V3+**) Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation [[Paper]](https://arxiv.org/abs/1802.02611) 
- [ ] (**GCN**) Large Kernel Matter, Improve Semantic Segmentation by Global Convolutional Network [[Paper]](https://arxiv.org/abs/1703.02719)
- [ ] (**UperNet**) Unified Perceptual Parsing for Scene Understanding [[Paper]](https://arxiv.org/abs/1807.10221)
- [ ] (**DUC, HDC**) Understanding Convolution for Semantic Segmentation [[Paper]](https://arxiv.org/abs/1702.08502) 
- [ ] (**PSPNet**) Pyramid Scene Parsing Network [[Paper]](http://jiaya.me/papers/PSPNet_cvpr17.pdf) 
- [ ] (**ENet**) A Deep Neural Network Architecture for Real-Time Semantic Segmentation [[Paper]](https://arxiv.org/abs/1606.02147)
- [x] (**U-Net**) Convolutional Networks for Biomedical Image Segmentation (2015): [[Paper]](https://arxiv.org/abs/1505.04597)
- [ ] (**SegNet**) A Deep ConvolutionalEncoder-Decoder Architecture for ImageSegmentation (2016): [[Paper]](https://arxiv.org/pdf/1511.00561)
- [x] (**FCN**) Fully Convolutional Networks for Semantic Segmentation (2015): [[Paper]](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) 

### Datasets
Various datasets, each has it own dataloader code.

- **CityScapes:** First download the images and the annotations (there is two types of annotations, Fine `gtFine_trainvaltest.zip` and Coarse `gtCoarse.zip` annotations, and the images `leftImg8bit_trainvaltest.zip`) from the official website [cityscapes-dataset.com](https://www.cityscapes-dataset.com/downloads/), extract all of them in the same folder, and use the location of this folder in `config.json` for training.
- **Sugar beet:**
- **Clover Weed:**
- **Our Synthetic Turfgrass dataset** https://www.zenodo.org/record/8030582
- **Our real Trufgrass: Release TBC**

## Train and Test
To Train the network use the following, note the configs folder contains all the configuration used.


```bash
python train.py --config ./configs/XXXXXX.json 
```

Resume on the `.pth` chekpoints.

```bash
python train.py --config config.json --resume ./saved/XXProjectXX/best_model.pth --eval true
```

tensorboard was stable in pyTorch 2 but it vey system dependant. 
Need to install tensorboard

```bash
tensorboard --logdir saved
```

### Data augmentation
Data augmentations are implemented here `\base\base_dataset.py`
- Note: class BaseDataSet(Dataset)
- Transforms.ColorJitter(brightness=0.xxx, contrast=0.xxx, saturation=0.xx, hue=0)]
- The current ROI in this repo is set to 400x400 pixels 
- Analysis is done outside of this repo on the data to select augmentation parms.
- Augmentation of the dataset are saved in augmentation folder see config files


## Code structure
The code structure is based on [pytorch-template](https://github.com/victoresque/pytorch-template/blob/master/README.md)

  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── trainer.py - the main trained
  │
  │
  ├── dataloader/ - loading the data for different segmentation datasets
  │
  ├── models/ - contains models for paper here
  │
  ├── configs/ - contains configs files for training and test for paper here
  │
  ├── saved/
  │   ├── runs/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   ├── base_dataset.py - All the data augmentations are implemented here
  │   └── base_trainer.py
  │  
  └── utils/ - small utility functions
      ├── losses.py - losses used in training the model
      ├── metrics.py - evaluation metrics used
      └── lr_scheduler - learning rate schedulers 
  ```
### ToDo List
- [x] Baseline pytorch import
- [x] Conda env setup with repo
- [x] Test pspnet, enet , deeplabv3 on cityscapes 
- [x] Add sugarbeet dataset pipeline
- [x] Add grassclover dataset pipeline 
- [x] Add synthetic dataset pipeline
- [x] Add real dataset pipeline
- [x] Google colab training tested and worked
- [x] conda env
- [ ] docker support with 

## Acknowledgement
- Code is based on [PyTorch-segmentation](https://github.com/yassouali/pytorch-segmentation)
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [Pytorch-Template](https://github.com/victoresque/pytorch-template/blob/master/README.m)
- [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

## Cite
```
 @inproceedings{IMVIP2023,
   author = {Stephen Foy and Simon Mc Loughlin},
   title = {Assessment of Synthetic Turfgrass Dataset Generation for Divot Detection},
   booktitle = { Irish Machine Vision and Image Processing Conference (IMVIP)},
   year = {2023}
  }
```


