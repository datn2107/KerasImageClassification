import importlib
from math import floor, ceil, log2
from typing import Tuple, List

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout

module_name = "tensorflow.kerascls.applications.{model_name}"
general_model_name = {"Xception": "xception",
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
    def __init__(self, model_name: str, num_class: int, backbone_weights: str = "imagenet",
                 trainable_backbone: bool = True, last_pooling_layer: str = "avg", num_dense: int = 0,
                 unit_first_dense_layer: int = 4096, activation_dense: str = 'relu',
                 units_remain_rate: float = 0.5, activation_last_dense: str = 'sigmoid',
                 dropout_layer: bool = True, dropout_rate: float = 0.3):
        self.model_name = model_name
        self.num_class = num_class
        self.last_pooling_layer = last_pooling_layer
        self.num_dense = num_dense
        self.unit_first_dense_layer = unit_first_dense_layer
        self.activation_dense = activation_dense
        self.units_remain_rate = units_remain_rate
        self.activation_last_dense = activation_last_dense
        self.dropout_layer = dropout_layer
        self.dropout_rate = dropout_rate

        # Load base model and their related function
        module = importlib.import_module(module_name.format(model_name=general_model_name[self.model_name]))
        self.base_model = getattr(module, self.model_name)(include_top=False, weights=backbone_weights,
                                                           pooling=last_pooling_layer)
        self.base_model.trainable = trainable_backbone
        self.preprocess_layer = getattr(module, "preprocess_input")
        self.decode_predictions = getattr(module, "decode_predictions")

    def _get_base_model_input_shape(self) -> Tuple[int, int, int]:
        model = self.base_model

        shape = model.input_shape
        height, weight, channel = shape[1], shape[2], shape[3]

        if height is None and weight is None:
            height = weight = 224
        return height, weight, channel

    def _get_base_model_output_shape(self) -> List:
        model = self.base_model
        return model.output_shape[1:]

    def _create_fully_connected_layer(self, backbone) -> tf.keras.layers.Layer:
        self.unit_first_dense_layer = min(self.unit_first_dense_layer, backbone.shape[1])
        max_dense = floor(log2(self.unit_first_dense_layer)) - ceil(log2(self.num_class))
        self.num_dense = min(self.num_dense, max_dense)

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

        if self.num_dense == 1:
            output = Dense(units=self.num_class, activation=self.activation_last_dense)(backbone)
        else:
            output = Dense(units=self.num_class, activation=self.activation_last_dense)(fc_layer[-1])
        return output

    def create_model_keras(self) -> tf.keras.Model:
        input_shape = self._get_base_model_input_shape()

        input_layer = tf.keras.Input(shape=input_shape)
        preprocess_layer = self.preprocess_layer(input_layer)
        base_model = self.base_model(preprocess_layer)

        if self.last_pooling_layer is None:
            flatten_layer = Flatten()(base_model)
            output = self._create_fully_connected_layer(backbone=flatten_layer)
        else:
            output = self._create_fully_connected_layer(backbone=base_model)

        return tf.keras.Model(input, output)
