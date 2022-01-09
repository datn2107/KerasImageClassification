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

```shell
$ pip install ./KerasImageClassification/requirements.txt
```

# Data Preparation

Create a directory contain `csv` file and `image` split into each folder.

**Note**: You can ignore test set

## Image Folder

```
dataroot 
|── train.csv
|── train
    |── *.png or *.jpeg
|── val.csv
|── val
    |── *.png or *.jpeg
|── test.csv (optional)
|── test (optional)
    |── *.png or *.jpeg

```

## Dataframe

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

The config file is (by default) in `configs/model.yaml`, which also a config file that `train.py` (by default) will use
to load model and other component. For more details you can see
in [here](https://github.com/datn2107/KerasImageClassification/blob/master/configs/CONFIG.md).

# Tutorial

## Display Training Information

This operation use to display config of model, optimizer, loss and metrics to verify your config file.

```shell
$ python display_training_info_py --model_config <config_path> 
```

You can also use `--num_classes` to specify the output of model

## Training

```shell
$ python train.py --data_root <data_directory> --batch_size <batch_size> --epoch <number_epoch>  
```


All the checkpoints of training process, new config file and result of training is saved (by default)
in `./saving_root`. You can change the `saving_root` by using `--siving_root <new_saving_root>`
argument. 

More particular your `saving_root` will save:
* `tf2`, `hdf5` model checkpoint in each epoch (optional)
* `tf2`, `hdf5` best model checkpoint
* `*.yaml` your model config file, but it contains checkpoint information to resume training easier 
* `log.csv` file csv contain loss and metrics point

**Note**: You need to ensure that `saving_root` doesn't contain a config file, or it contains a config file but in a
different name, because the new config file will be moved into `saving_root`.

You specify config file by using `--model_config <model_config_path>`, if not (by default) it will use the config file
in `./configs/model.cfg` to load model and checkpoint.

You also can add `--save_best_only` to save the best checkpoint only.


