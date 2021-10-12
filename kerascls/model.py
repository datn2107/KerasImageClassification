import os
import importlib
import warnings
from math import floor, ceil, log2
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout

# These constant use to define model of tf.keras.applications when we only have model name at string type
MODULE_NAME_FORMAT = "tensorflow.keras.applications.{model_name}"
MODULE_NAME = {"Xception": "xception",
               "VGG16": "vgg16", "VGG19": "vgg19",
               "ResNet50": "resnet", "ResNet101": "resnet", "ResNet152": "resnet",
               "ResNet50V2": "resnet_v2", "ResNet101V2": "resnet_v2", "ResNet152V2": "resnet_v2",
               "InceptionV3": "inception_v3", "InceptionResNetV2": "inception_resnet_v2",
               "MobileNet": "mobilenet", "MobileNetV2": "mobilenet_v2",
               "DenseNet121": "densenet", "DenseNet169": "densenet", "DenseNet201": "densenet",
               "NASNetMobile": "nasnet", "NASNetLarge": "nasnet",
               "EfficientNetB0": "efficientnet", "EfficientNetB1": "efficientnet",
               "EfficientNetB2": "efficientnet", "EfficientNetB3": "efficientnet",
               "EfficientNetB4": "efficientnet", "EfficientNetB5": "efficientnet",
               "EfficientNetB6": "efficientnet", "EfficientNetB7": "efficientnet", }


class KerasModel:
    def __init__(self, model_name: str, num_class: int, input_shape: Tuple[int, int, int] = (224, 224, 3),
                 backbone_weights: str = "imagenet", trainable_backbone: bool = True, last_pooling_layer: str = "avg",
                 num_dense: int = 1, unit_first_dense_layer: int = 4096, activation_dense: str = 'relu',
                 units_remain_rate: float = 0.5, activation_last_dense: str = 'sigmoid', dropout_layer: bool = True,
                 dropout_rate: float = 0.3, **ignore):
        self.full_model = None
        self.num_class = num_class
        self.input_shape = input_shape
        self.last_pooling_layer = last_pooling_layer
        self.num_dense = num_dense
        self.unit_first_dense_layer = unit_first_dense_layer
        self.activation_dense = activation_dense
        self.units_remain_rate = units_remain_rate
        self.activation_last_dense = activation_last_dense
        self.dropout_layer = dropout_layer
        self.dropout_rate = dropout_rate

        # Each specific model is a function of module in tf.keras.applications
        # to load model we need to load module that contain it
        module = importlib.import_module(MODULE_NAME_FORMAT.format(model_name=MODULE_NAME[model_name]))
        self.backbone = getattr(module, model_name)(include_top=False, weights=backbone_weights,
                                                    pooling=last_pooling_layer)
        self.backbone.trainable = trainable_backbone
        # Each model in tf.keras.applications will have different requirement about image input
        # Some need to normalization some not, so we need to use their preprocess_input function to handle it
        self.preprocess_layer = getattr(module, "preprocess_input")

    def _create_fully_connected_layers(self, backbone) -> tf.keras.layers.Layer:
        # Guarantee the unit of first dense layer fit with backbone output
        self.unit_first_dense_layer = min(self.unit_first_dense_layer, backbone.shape[1])
        # If the dense layer is too much the model will be broken (unit last dense is smaller than output unit)
        max_dense = floor(log2(self.unit_first_dense_layer)) - ceil(log2(self.num_class))
        self.num_dense = min(self.num_dense, max_dense)

        # Create Dense layers
        fc_layer = []
        if self.num_dense > 1:
            fc_layer.append(Dense(units=self.unit_first_dense_layer, activation=self.activation_dense)(backbone))
            if self.dropout_layer:
                fc_layer.append(Dropout(rate=self.dropout_rate)(fc_layer[-1]))

        for _ in range(2, self.num_dense):
            fc_layer.append(
                Dense(units=fc_layer[-1].shape[1] * self.units_remain_rate, activation=self.activation_dense)(
                    fc_layer[-1]))
            if self.dropout_layer:
                fc_layer.append(Dropout(rate=self.dropout_rate)(fc_layer[-1]))

        # Create output layer
        if self.num_dense == 1:
            # num_dense == 1 it mean these model don't need any dense layer
            output = Dense(units=self.num_class, activation=self.activation_last_dense)(backbone)
        else:
            output = Dense(units=self.num_class, activation=self.activation_last_dense)(fc_layer[-1])
        return output

    def create_model_keras(self) -> tf.keras.Model:
        input_layer = tf.keras.Input(shape=self.input_shape)
        preprocess_layer = self.preprocess_layer(input_layer)
        backbone = self.backbone(preprocess_layer)

        if self.last_pooling_layer is None:
            flatten_layer = Flatten()(backbone)
            output = self._create_fully_connected_layers(backbone=flatten_layer)
        else:
            output = self._create_fully_connected_layers(backbone=backbone)

        self.full_model = tf.keras.Model(input_layer, output)

    def load_checkpoint_weight(self, weights_cp_path=None, weights_cp_dir=None):
        def load_latest_weight():
            latest_checkpoint = None
            for checkpoint in os.listdir(weights_cp_dir):
                checkpoint = os.path.join(weights_cp_dir, checkpoint)
                # Format of checkpoint of tensorflow always has 2 file with the prefix:
                #   *.index
                #   *.data-xxxxx-of-xxxxx (x is a number)
                # The xxxxx is change depend on your system so use .index to identify each checkpoint in directory
                if checkpoint.endswith(".index"):
                    if latest_checkpoint is None:
                        latest_checkpoint = checkpoint
                    elif os.path.getmtime(latest_checkpoint) < os.path.getmtime(checkpoint):
                        latest_checkpoint = checkpoint
            return ".".join(latest_checkpoint.split('.')[:-1])

        if weights_cp_path is not None or weights_cp_dir is not None:
            if weights_cp_dir is not None:
                weights_cp_path = load_latest_weight()
            self.full_model.load_weights(weights_cp_path)
            print("Load checkpoints from ", weights_cp_path)
        else:
            warnings.warn("There are no weights checkpoint to load.")

    def get_backbone_weight(self):
        pass
