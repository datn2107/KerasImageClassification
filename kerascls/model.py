import importlib
import os
import warnings
from typing import Tuple, List, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout

from .loss_and_metric import load_loss, load_optimizer, load_list_metric

# These constant use to define model of tf.keras.applications when we only have model name at string type
MODULE_NAME_FORMAT = "tensorflow.keras.applications.{model_name}"
MODULE_NAME = {"Xception": "xception",
               "VGG16": "vgg16", "VGG19": "vgg19",
               "ResNet50": "resnet", "ResNet101": "resnet", "ResNet152": "resnet",
               "ResNet50V2": "resnet_v2", "ResNet101V2": "resnet_v2", "ResNet152V2": "resnet_v2",
               "InceptionV3": "inception_v3", "InceptionResNetV2": "inception_resnet_v2",
               "MobileNet": "mobilenet", "MobileNetV2": "mobilenet_v2",
               "MobileNetV3Small": "mobilenet_v3", "MobileNetV3Large": "mobilenet_v3",
               "DenseNet121": "densenet", "DenseNet169": "densenet", "DenseNet201": "densenet",
               "NASNetMobile": "nasnet", "NASNetLarge": "nasnet",
               "EfficientNetB0": "efficientnet", "EfficientNetB1": "efficientnet",
               "EfficientNetB2": "efficientnet", "EfficientNetB3": "efficientnet",
               "EfficientNetB4": "efficientnet", "EfficientNetB5": "efficientnet",
               "EfficientNetB6": "efficientnet", "EfficientNetB7": "efficientnet", }


class KerasModelGenerator:
    """This class support you to create image classification model with backbone of tf.keras.applications

        Base model:
            - All the allowed base model is in tf.application.keras
              [https://www.tensorflow.org/api_docs/python/tf/keras/applications]
            - This model contains the pre-processing layer of correspond basemodel, so it didn't need normalized data
            - You can load weight to backbone by pass your own checkpoint path
                (The file of checkpoint need to match with the format of tensorflow's weight checkpoints)
                [https://www.tensorflow.org/tutorials/keras/save_and_load#checkpoint_callback_options]

        Fully connected layer:
            - This will create fully connected layer base on the number of dense layer, activation of them.
                You also can choose the remained unit fraction after each layer
            - It also can add dropout layer before each layer

        Attribution:
            - backbone (tf.keras.Model): Backbone Model
            - full_model (tf.keras.Model): Full Model create by combination of backbone and fully connected layers
            - create_full_model: Function use to create full_model and return it
            - load_weight: Use to load weight into full_model
            - compile: Compile full model with optimizer, loss and metrics
            - detect: Detect image and return result of it
    """

    def __init__(self, model_name: str, height: int = 244, width: int = 244,
                 backbone_weights: str = "imagenet", trainable_backbone: bool = True, last_pooling_layer: str = None,
                 num_dense: int = 1, units_first_dense_layer: int = 4096, activation_dense: str = 'relu',
                 remained_units_fraction: float = 0.5, activation_last_dense: str = 'sigmoid',
                 dropout_layer: bool = True, dropout_rate: float = 0.3, **ignore):
        """
        :param model_name (str): Name of base model (backbone)
                                 [https://www.tensorflow.org/api_docs/python/tf/keras/applications]
        :param num_classes (int): Number of classes to classify images
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
        :param dense_activation (str): The activation of each dense layer
        :param remained_units_fraction (float): The remain unit after each dense layer
        :param last_dense_activation (str): This is the activation of the output layers
        :param dropout_layer (bool): Add the dropout layer or not
        :param dropout_rate (float): The dropout rate of each dropout layer
        :param ignore: This use to ignore every argument that don't allow in this class
                       While you passing the dict to function (**dict)
        """
        self.full_model = None
        self.input_shape = (height, width, 3)
        self.last_pooling_layer = last_pooling_layer
        self.num_dense = num_dense
        self.unit_first_dense_layer = units_first_dense_layer
        self.dense_activation = activation_dense
        self.remained_units_fraction = remained_units_fraction
        self.last_dense_activation = activation_last_dense
        self.dropout_layer = dropout_layer
        self.dropout_rate = dropout_rate

        # Each specific model is a function of module in tf.keras.applications
        # to load model we need to load module that contain it
        model_module = importlib.import_module(MODULE_NAME_FORMAT.format(model_name=MODULE_NAME[model_name]))
        self.backbone = getattr(model_module, model_name)(include_top=False, weights=backbone_weights,
                                                          pooling=last_pooling_layer)
        self.backbone.trainable = trainable_backbone

        # Each model in tf.keras.applications will have different requirement about image input
        # Some need to normalization some not, so we need to use their preprocess_input function to handle it
        self.preprocess_layer = getattr(model_module, "preprocess_input")

    def _create_fully_connected_layers(self, num_classes: int, backbone: tf.keras.Model) -> tf.keras.layers.Layer:
        """Create fully connected layers and add to backbone model"""

        fc_layers = [backbone]
        if self.dropout_layer:
            fc_layers = [Dropout(rate=self.dropout_rate)(backbone)]

        # Ensure the unit of first dense layer smaller backbone output
        # because each backbone each last pooling layer will output different number of unit
        self.unit_first_dense_layer = min(self.unit_first_dense_layer, fc_layers[-1].shape[1])

        if self.num_dense > 1:
            fc_layers.append(Dense(units=self.unit_first_dense_layer, activation=self.dense_activation)(fc_layers[-1]))
            if self.dropout_layer:
                fc_layers.append(Dropout(rate=self.dropout_rate)(fc_layers[-1]))

        for _ in range(2, self.num_dense):
            if int(fc_layers[-1].shape[1] * self.remained_units_fraction) < num_classes:
                break
            fc_layers.append(
                Dense(units=int(fc_layers[-1].shape[1] * self.remained_units_fraction), activation=self.dense_activation)(
                    fc_layers[-1]))
            if self.dropout_layer:
                fc_layers.append(Dropout(rate=self.dropout_rate)(fc_layers[-1]))

        # Create output layer
        fc_layers.append(Dense(units=num_classes, activation=self.dense_activation)(fc_layers[-1]))
        output = Dense(units=num_classes, activation=self.last_dense_activation)(fc_layers[-1])
        return output

    def create_model(self, num_classes):
        """Combine backbone model and fully connected layers to create full model"""

        input_layer = tf.keras.Input(shape=self.input_shape)
        preprocess_layer = self.preprocess_layer(input_layer)
        backbone = self.backbone(preprocess_layer)

        if self.last_pooling_layer is None:
            flatten_layer = Flatten()(backbone)
            print(flatten_layer.shape[-1])
            output = self._create_fully_connected_layers(num_classes=num_classes, backbone=flatten_layer)
        else:
            output = self._create_fully_connected_layers(num_classes=num_classes, backbone=backbone)

        self.full_model = tf.keras.Model(input_layer, output)
        return self.full_model

    def load_weights(self, weights_cp_path=None, weights_cp_root=None, **ignore):
        """Load checkpoint to model

        :param weights_cp_path: Weights checkpoint path for model
        :param weights_cp_root: Directory contain weights checkpoint for model, it will automatic get the latest modify
                               Load weight checkpoint will consider first before weights checkpoint path
        :param ignore: Use to ignore undesired arguments while you passing by dict to function (**dict)
        """

        def get_latest_checkpoint():
            """Load the latest checkpoint in directory which latest determine depend on the modify time"""
            latest_checkpoint = None
            for checkpoint in os.listdir(weights_cp_root):
                checkpoint = os.path.join(weights_cp_root, checkpoint)
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

        if weights_cp_path is not None or weights_cp_root is not None:
            if weights_cp_root is not None:
                weights_cp_path = get_latest_checkpoint()
            self.full_model.load_weights(weights_cp_path)
            print("Load checkpoints from ", weights_cp_path)
        else:
            warnings.warn("There are no weights checkpoint to load.")

        return self.full_model

    def compile_model(self, loss_config: Dict, optimizer_config: Dict, metrics_config: List[Dict]):
        """ Compile full model

        :param loss_config: Allowed key
                            - optimizer (str): Optimizer Name
                            + Additional it can contain the parameters of optimizer
        :param optimizer_config: Allowed key
                                - loss (str): Loss Name (Default: BinaryCrossentropy)
                                + Additional it can contain the parameters of loss
        :param metrics_config: List of dict which have allowed key
                                - metric (str): Metric Name (Default: BinaryAccuracy)
                                + Additional it can contain the parameters of metric
        """
        self.full_model.compile(optimizer=load_optimizer(**optimizer_config),
                                loss=load_loss(**loss_config),
                                metrics=load_list_metric(metrics_config))

        return self.full_model

    def detect(self, image_path: str) -> np.array:
        """Use model to detect image and return numpy array contain probability of each class"""

        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False, dtype=tf.float32)
        image = tf.image.resize(image, (self.input_shape[0], self.input_shape[1]))
        # Add "batch" dimension because model require tensor shape (None, input_shape[0], input_shape[1], 3)
        image = tf.expand_dims(image, axis=0)

        return tf.squeeze(self.full_model(image)).numpy()


if __name__ == '__main__':
    from config import ModelConfigReader

    config_reader = ModelConfigReader(config_path="../configs/model.yaml")
    model_config = config_reader.get_model_config()
    print(model_config)

    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    model_generator = KerasModelGenerator(**model_config)
    model = model_generator.create_model(num_classes=10)
    print(model.summary())

    # for layer in model.layers:
    #     print(layer)
