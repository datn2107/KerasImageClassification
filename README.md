# Introduction

This is the toolbox to build keras image classification model. Which can help you build a quick and dirty model for
image classification

# Feature

* Easy to build a quick and dirty image classification model
* Contain all available models, optimizers, losses and metrics in TensorFlow Keras
    * [Models](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
    * [Optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
    * [Losses](https://www.tensorflow.org/api_docs/python/tf/keras/losses)
    * [Metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)
* Custom fully connected layer through config
* Easy to config model, optimizer, loss and metrics

# Model Zoo 

You can build classification model has all backbone model in 
[tf.keras.application](https://www.tensorflow.org/api_docs/python/tf/keras/applications#functions)

# Installation

```shell
$ git clone https://github.com/datn2107/KerasImageClassification.git .
```

## Requirements
* tensorflow >= 2.6
* pandas >= 1.1
* numpy >= 1.19
* matplotlib >= 3.4
* sklearn

```shell
$ pip install ./KerasImageClassification/requirements.txt
```

## Install KerasImageClassification

```shell
$ pip install ./KerasImageClassification
```

# Data Preparation

## Image Folder
Create a directory contain all image
```
image data root (folder containing all images)
|── All image files 
```

## Dataframe
Create a dataframe `.csv` contain metadata of all image
```
    filename    | Class 1 | Class 2 | Class3
----------------|---------|---------|-------- 
*.png or *.jpeg |    1    |    1    |    0
      ....      |   ...   |   ...   |   ...
```

## Config

The config file use to setting model, optimizer, loss and metric is (by default) in `configs/setting.cfg`. 
For more details you can see in [here](https://github.com/datn2107/KerasImageClassification/tree/master/configs).

# Tutorial

## Model Summary
```shell
$ python /content/KerasImageClassification/tools/model_summary.py
```

Display summary of your model, you can use your own model config by using `--config <config_path>`.

## Optimizer, Loss, and Metrics Config 
```shell
$ python /content/KerasImageClassification/tools/model_summary.py
```

Display config of optimizer, loss and metrics, you can use your own config by using `--config <config_path>`.

## Training
```shell
$ python KerasImageClassification/tools/train.py --batch 64 --epoch 30 --image_dir '' --metadata_path '' 
```
All the checkpoints of training process, result of training is saved (by default) in `saving_dir`. 
You can change the `saving_dir` by using `--siving_dir <new_saving_dir>` argument

You also can change fraction of training, validation and test by unsing `--train_size`, `--val_size` and `--test_size` 
by default is `--train_size 0.7 --val_size 0.15 --test_size 0.15` (you need to guarantee that sum of all fraction equal 1)
