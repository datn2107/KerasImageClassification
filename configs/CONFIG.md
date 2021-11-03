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

This part is used to change the input shape of model, if you leave them equal `None` or not declare them by default it
will equal `(224, 224)`.

Note that some models
of [tf.keras.application](https://www.tensorflow.org/api_docs/python/tf/keras/applications#functions) only have weight
for some specific input shape, before initialize them you need to check out the model document first.

```buildoutcfg
[Base Model]
model_name = ResNet50V2
backbone_weights = imagenet
trainable_backbone = True
last_pooling_layer = avg
```

You can get model name and their parameters
in [tf.keras.application](https://www.tensorflow.org/api_docs/python/tf/keras/applications#functions).

You can specify your own backbone weight by pass the checkpoint path into `backbone_weights`, and you can also choose to
freeze backbone or not.

```buildoutcfg
[Fully Connected Layer]
num_dense = 3
units_first_dense_layer = 4096
units_remain_fraction = 0.7
activation_dense = relu
activation_last_dense = sigmoid
dropout_layer = True
dropout_rate = 0.3
```

Every layer in Fully Connected Layer
use [tf.keras.layers.Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense).

The number of dense layers is the number of fully connected layers, which you need to ensure is greater than 1.

N-1 first layers will have the same activation, and their units will decrease by one specific fraction (which you will
specify by define the `units_remain_fraction`). You just only need to define the `units_first_dense_layer` every other
layer will automatically define by `units_remain_fraction` base one the `units_first_dense_layer`.

With the last dense layer, which is called output layer, You only need to define the activation of it, the units of it
is equal the number class of data.

You also can add the dropout layer (with specific `dropout_rate`) after each dense layer.

## Checkpoint

```buildoutcfg
[Checkpoints]
weights_cp_dir = None
weights_cp_path = None
best_weights_cp_path = None
last_epoch = 0
```

There are 2 ways to load weight into model.

* `weights_cp_dir` use to define the directory path which contain checkpoint weights of model
  (the latest modify checkpoint will be selected) (Load first if it is not None)
* `weights_cp_path` use to define the checkpoint path (if there are no weights_cp_dir it will load weights_cp_path)

The last 2 parameters are used to resume training

* `best_weights_cp_path` is use to load the loss of the bested checkpoint already have in last training precess. Without
  it, the best checkpoint of whole training process is just the best of the latest process.
* `last_epoch` is the last epoch of the last training process

## Optimizer, Loss, and Metrics

```buildoutcfg
[Optimizer]
optimizer = Adam

[Loss]
loss = BinaryCrossentropy

[Metric]
metric = BinaryAccuracy
```

You can find optimizer, loss and metrics, and their parameters in these link
[optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers#classes),
[loss](https://www.tensorflow.org/api_docs/python/tf/keras/losses#classes_2),
[metric](https://www.tensorflow.org/api_docs/python/tf/keras/metrics#classes)

You can add parameters for each optimizer, loss and metric by define these parameters in its section. Example (similar
to optimizer and loss):

```buildoutcfg
[Optimizer]
optimizer = Adam
learning_rate = 0.0001
beta_2 = 0.998
```

You also can define more than one metric by create new section with prefix is 'Metric'. Example:
```buildoutcfg
[Metric]
metric = BinaryAccuracy

[Metric_cross]
metric = BinaryCrossentropy

[Metric1] 
metric = CosineSimilarity
```