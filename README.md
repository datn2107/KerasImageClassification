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
data_root (folder containing all images)
|── *.png or *.jpeg
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

The usage of config file:

* Config model
* Specify the path of model's weights
* Config optimizer
* Config loss
* Config metrics

The config file is (by default) in `configs/setting.cfg`, which also a config file that `train.py` and `eval.py` (by
default) will use to load model. For more details you can see
in [here](https://github.com/datn2107/KerasImageClassification/blob/master/configs/CONFIG.md).

# Tutorial

## Model Summary

```shell
$ python /content/KerasImageClassification/tools/model_summary.py
```

Display summary of your model, you can use your own model config by using `--config <config_path>`.

## Optimizer, Loss, and Metrics Config

```shell
$ python /content/KerasImageClassification/tools/training_arg.py
```

Display config of optimizer, loss and metrics, you can use your own config by using `--config <config_path>`.

## Training

```shell
$ python KerasImageClassification/tools/train.py --batch 64 --epoch 30 --image_dir <data_root> --metadata_path <metadata_path> --config <config_path>
```

All the checkpoints of training process, result of training is saved (by default)
in `KerasImageClassification/saving_dir`. You can change the `saving_dir` by using `--siving_dir <new_saving_dir>`
argument.

**Note that**: You need to ensure that `saving_dir` doesn't contain config file, because the new config file will be
moved into `saving_dir` and it will contain your model config and checkpoint information to resume training easier.

You also can change fraction of training, validation and test by using `--train_size`, `--val_size` and `--test_size`
by default is `--train_size 0.7 --val_size 0.15 --test_size 0.15` (you need to ensure that sum of all fraction equal 1)

## Evaluation

```shell
$ python KerasImageClassification/tools/eval.py --image_dir <data_root> --metadata_path <metadata_path> --config <config_path>
```

All data referred by metadata file will be used to evaluate.

To using this you need to specify config or model you want to use, and in that config file the `best_weights_cp_path` is
not None or not define. The weight of model will load from this file, so that you also need to ensure that the model in
config file is similar to the weights in `best_weights_cp_path`.

By default, the result of this evaluation is saved in `KerasImageClassification/saving_dir`. You can change
the `saving_dir` by using `--siving_dir <new_saving_dir>`
argument.