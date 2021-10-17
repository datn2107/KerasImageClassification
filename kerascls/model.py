import os
import importlib
import warnings
from math import floor, ceil, log2
from typing import Tuple, List, Dict

import numpy
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout

from kerascls.loss_and_metric import load_loss, load_optimizer, load_list_metric

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
    """This class support you to create image classification model with backbone of tf.keras.applications

        Argumentation:
            - This model also have 2 argumentation layers:
                + random flip (horizontal and vertical)
                + random rotate (3/10 pi)

        Base model:
            - All the allowed base model is in tf.application.keras
              [https://www.tensorflow.org/api_docs/python/tf/keras/applications]
            - This model will contain the pre-processing layer of the base model so it didn't need to normalized data
            - You can load weight to backbone by pass your own checkpoint path
                (The file of checkpoint need to match with the format of tensorflow's weight checkpoints)
                [https://www.tensorflow.org/tutorials/keras/save_and_load#checkpoint_callback_options]

        Fully connected layer:
            - This will create fully connected layer base on the number of dense layer, activation of them.
                You also can choose the remain unit fraction after each layer
            - It also can add dropout layer before each layer

        Attribution:
            - backbone (tf.keras.Model): Backbone Model
            - full_model (tf.keras.Model): Full Model create by combination of backbone and fully connected layers
            - create_full_model: Function use to create full_model and return it
            - load_weight: Use to load weight into full_model
            - compile: Compile full model with optimizer, loss and metrics
            - detect: Detect image and return result of it
    """

    def __init__(self, model_name: str, num_class: int, input_shape: Tuple[int, int, int] = (224, 224, 3),
                 backbone_weights: str = "imagenet", trainable_backbone: bool = True, last_pooling_layer: str = "avg",
                 num_dense: int = 1, unit_first_dense_layer: int = 4096, activation_dense: str = 'relu',
                 units_remain_fraction: float = 0.5, activation_last_dense: str = 'sigmoid', dropout_layer: bool = True,
                 dropout_rate: float = 0.3, **ignore):
        """
        :param model_name (str): Name of base model (backbone)
                                 [https://www.tensorflow.org/api_docs/python/tf/keras/applications]
        :param num_class (int): Number of classes to classify images
        :param input_shape (tuple): Input shape will pass to model
                                    It need to guarantee that the number of channel is equal 3
        :param backbone_weights (str): Weight for your backbone model
                                       There are 3 option 'imagenet', None or path of checkpoint weight
        :param trainable_backbone (bool): Allowed to train backbone or not
        :param last_pooling_layer (str): The last pooling layer of backbone
                                         None: It automatically add a flatten layer before pass to fully connected layer
                                         'avg': The global average pooling will be apply
                                         'max': The global max pooling will be apply
        :param num_dense (int): Number of dense layer in fully connected layers
        :param unit_first_dense_layer (int): The number unit of the first dense layer
        :param activation_dense (str): The activation of each dense layer
        :param units_remain_fraction (float): The remain unit after each dense layer
        :param activation_last_dense (str): This is the activation of the output layers
        :param dropout_layer (bool): Add the dropout layer or not
        :param dropout_rate (float): The dropout rate of each dropout layer
        :param ignore: This use to ignore every argument that don't allow in this class
                       While you passing the dict to function (**dict)
        """

        if input_shape[2] != 3:
            raise ValueError('The number of channel must be 3, its must be RGB image.')

        self.full_model = None
        self.num_class = num_class
        self.input_shape = input_shape
        self.last_pooling_layer = last_pooling_layer
        self.num_dense = num_dense
        self.unit_first_dense_layer = unit_first_dense_layer
        self.activation_dense = activation_dense
        self.units_remain_fraction = units_remain_fraction
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

    def _create_fully_connected_layers(self, backbone: tf.keras.Model) -> tf.keras.layers.Layer:
        """Create fully connected layers and add to backbone model"""

        # Force that the unit of first dense layer smaller backbone output
        # because each backbone each last pooling layer will output different number of unit
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

        # only create num_dense-1 layers because 1 is for output layer
        for _ in range(2, self.num_dense):
            fc_layer.append(
                Dense(units=fc_layer[-1].shape[1] * self.units_remain_fraction, activation=self.activation_dense)(
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

    def create_full_model(self):
        """Combine backbone model and fully connected layers to create full model"""

        input_layer = tf.keras.Input(shape=self.input_shape)
        # Add argumentation layer
        flip_layer = tf.keras.layers.RandomFlip()(input_layer)
        rotate_layer = tf.keras.layers.RandomRotation(0.3)(flip_layer)
        preprocess_layer = self.preprocess_layer(rotate_layer)
        backbone = self.backbone(preprocess_layer)

        if self.last_pooling_layer is None:
            flatten_layer = Flatten()(backbone)
            output = self._create_fully_connected_layers(backbone=flatten_layer)
        else:
            output = self._create_fully_connected_layers(backbone=backbone)

        self.full_model = tf.keras.Model(input_layer, output)
        return self.full_model

    def get_backbone_weight(self):
        pass

    def load_weights(self, weights_cp_path=None, weights_cp_dir=None, **ignore):
        """Load checkpoint to model

        :param weights_cp_path: Weights checkpoint path for model
        :param weights_cp_dir: Directory contain weights checkpoint for model, it will automatic get the latest modify
                               Load weight checkpoint will consider first before weights checkpoint path
        :param ignore: Use to ignore undesired arguments while you passing by dict to function (**dict)
        """

        def load_latest_weight():
            """Load the latest checkpoint in directory which latest determine depend on the modify time"""
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

    def compile(self, loss_info: Dict, optimizer_info: Dict, metrics_info: List[Dict]):
        """ Compile full model

        :param loss_info: Allowed key
                            - optimizer (str): Optimizer Name
                            + Additional it can contain the parameters of optimizer
        :param optimizer_info: Allowed key
                                - loss (str): Loss Name (Default: BinaryCrossentropy)
                                + Additional it can contain the parameters of loss
        :param metrics_info: List of dict which have allowed key
                                - metric (str): Metric Name (Default: BinaryAccuracy)
                                + Additional it can contain the parameters of metric
        """
        self.full_model.compile(optimizer=load_optimizer(**optimizer_info),
                                loss=load_loss(**loss_info),
                                metrics=load_list_metric(metrics_info))

    def detect(self, image_path: str) -> numpy.array:
        """Use model to detect image and return numpy array contain probability of each class"""

        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False, dtype=tf.float32)
        image = tf.image.resize(image, (self.input_shape[0], self.input_shape[1]))
        # Add "batch" dimension because model require tensor shape (None, input_shape[0], input_shape[1], 3)
        image = tf.expand_dims(image, axis=0)

        return tf.squeeze(self.full_model(image)).numpy()
