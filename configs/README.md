## Model Config

There are 3 part in config file which use to build model:

* Input shape
* Base Model
* Fully Connected Layer

```buildoutcfg
[Input Shape]
height = None
width = None
```

```buildoutcfg
[Base Model]
model_name = ResNet50V2
backbone_weights = imagenet
trainable_backbone = True
last_pooling_layer = avg
```

```buildoutcfg
[Fully Connected Layer]
num_dense = 3
unit_first_dense_layer = 4096
units_remain_fraction = 0.7
activation_dense = relu
activation_last_dense = sigmoid
dropout_layer = True
dropout_rate = 0.3
```

## Checkpoint

```buildoutcfg
[Checkpoints]
weights_cp_dir = None
weights_cp_path = None
best_weights_cp_path = None
last_epoch = 0
```

There are 2 ways to load checkpoint into model.

* `weights_cp_dir` use to define the directory path which contain checkpoint weights of model
  (the latest modify checkpoint will be selected) (Load first if it is not None)
* `weights_cp_path` use to define the checkpoint path (if there are no weights_cp_dir it will load weights_cp_path)

The last 2 parameters are used to resume training

* `best_weights_cp_path` is use to load the loss of the bested checkpoint already have in last training precess. Without
  it, the best checkpoint of whole training process is just the best of the latest process.
* `last_epoch` is the last epoch of the last training process