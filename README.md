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

# Data Preparation

## Dataset

### Image Folder
Create a directory contain all image
```
image data root (folder containing all images)
|── All image files 
```

### Dataframe
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

## Training
```shell
$ python KerasImageClassification/tools/train.py --batch 64 --epoch 30 --image_dir '' --metadata_path '' 
```
All the checkpoints of training process, result of training is saved (by default) in `saving_dir`. 
You can change the `saving_dir` by using `--siving_dir <new_saving_dir>` argument

You also can change fraction of training, validation and test by unsing `--train_size`, `--val_size` and `--test_size` 
by default is `--train_size 0.7 --val_size 0.15 --test_size 0.15` (you need to guarantee that sum of all fraction equal 1)

