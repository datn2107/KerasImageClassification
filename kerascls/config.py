import warnings
from typing import List, Dict, Any

import yaml


class ConfigReader:
    """Read data from config and return to dict for each section

            Allowed section:
                - Checkpoint: Contain path to checkpoint
                - Input Shape: Contain size of the image size to resize
                - Base Model: Contain the argument of the base model
                - Fully Connected Layer: Contain the config for fully connected layer
                - Optimizer: Contain name of optimizer and arguments for its
                - Loss: Contain name of loss and arguments for its
                - Metric*: * is mean you can create more than 1 section with prefix string is "Metric"
                           Each section of this will contain name of metric and its arguments
    """
    def __init__(self, config_path: str):
        with open(config_path, 'r') as stream:
            self.config_data = yaml.safe_load(stream)

    @classmethod
    def _fill_missing_key(cls, config_dict: Dict, allow_keys: List[str]) -> Dict:
        """Assign None to the missing arguments"""

        # Guarantee there are no missing arguments
        for key in allow_keys:
            if key not in config_dict:
                config_dict[key] = None
        return config_dict

    def get_checkpoint_config(self) -> Dict:
        """
        :return: Dict contain:
                - weights_cp_dir (str): Path of director which contain weights checkpoints for model
                - weights_cp_path (str): Path of weights checkpoint for model
                - best_weights_cp_path (str): Path of best weights checkpoint for model
                - last_epoch (int): The epoch of the last training
        """
        # Set None value instead of missing arguments
        # because our function can handle None value
        checkpoint_config = self.config_data['Checkpoints']
        allow_keys = ['weights_cp_dir', 'weights_cp_path', 'best_weights_cp_path']
        checkpoint_config = self._fill_missing_key(checkpoint_config, allow_keys)
        if checkpoint_config.get('last_epoch', None) is None:
            self.config_data['Checkpoints']['last_epoch'] = 0
        return checkpoint_config

    def get_model_config(self) -> Dict:
        """
        :return: Dict contain:
                - input_shape (int): Input shape which will pass to model

                - model_name (str): Name of base (backbone) model (NOT NULL)
                - backbone_weights (str): Weight for your backbone model
                - trainable_backbone (bool): Trainable backbone or not
                - last_pooling_layer (str): The last layer of backbone model

                - num_dense (int): Number of dense layer of fully connected layers
                - units_first_dense_layer (int): Number units in each dense layer
                - units_remain_fraction (float): The fraction of remain unit
                - activation_dense (str): The activation of each dense layer
                - activation_last_dense (str): The activation of the last dense layer or the output layer
                - dropout_layer (bool): Add dropout after each dense layer or not
                - dropout_rate (float): Dropout rate of each dropout layer
        """
        height = self.config_data['InputShape']['height']
        width = self.config_data['InputShape']['height']
        input_config = {'input_shape': (height, width, 3)}
        model_config = self.config_data['BaseModel']
        fcl_config = self.config_data['FullyConnectedLayer']

        full_model_config = {**input_config, **model_config, **fcl_config}

        # Only need to guarantee the model_name and num_dense are not missing or not None
        # because other arguments is has default value in build model class
        if full_model_config.get('model_name', None) is None:
            raise(ValueError, 'Missing model name in config')

        # guarantee that num_dense not less than 1
        if full_model_config.get('num_dense', None) is not None \
                and full_model_config.get('num_dense') <= 0:
            warnings.warn("Number dense layer is less than 1. It will set to the default num_dense = 1")
            full_model_config['num_dense'] = 1

        # Eliminate parameter has None value (except last_pooling_layer, None is accepted)
        # because this will pass to function by **dict
        # and parameters in function is already has default value
        # so if it has None value it will broke default value of function
        if full_model_config['input_shape'][0] is None or full_model_config['input_shape'][1] is None:
            full_model_config.pop('input_shape')
        keys = ['backbone_weights', 'trainable_backbone', 'num_dense',
                'unit_first_dense_layer', 'units_remain_fraction',
                'activation_dense', 'activation_last_dense', 'dropout_layer', 'dropout_rate']
        for key in keys:
            if key in full_model_config and full_model_config.get(key) is None:
                full_model_config.pop(key)

        return full_model_config

    def get_optimizer_config(self) -> Dict:
        """
        :return: Dict contain:
                - optimizer (str): Optimizer Name (Default: SGD)
                + Additional it can contain the parameters of optimizer
        """
        optimizer_config = self.config_data['Optimizer']
        if 'optimizer' not in optimizer_config:
            optimizer_config['optimizer'] = 'SGD'
        return optimizer_config

    def get_loss_config(self) -> Dict:
        """
        :return: Dict contain:
                - loss (str): Loss Name (Default: BinaryCrossentropy)
                + Additional it can contain the parameters of loss
        """
        loss_config = self.config_data['Loss']
        if 'loss' not in loss_config:
            loss_config['loss'] = 'BinaryCrossentropy'
        return loss_config

    def get_list_metric_config(self) -> List[Dict]:
        """
        :return: List of dict which each contain:
                - metric (str): Metric Name (Default: BinaryAccuracy)
                + Additional it can contain the parameters of metric
        """
        list_metric_config = []

        for section in self.config_data.keys():
            if not section.startswith("Metric"):
                continue
            metric_config = self.config_data[section]
            if 'metric' not in metric_config:
                metric_config = 'BinaryAccuracy'
            list_metric_config.append(metric_config)
        return list_metric_config


if __name__ == '__main__':
    import os
    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config_path = os.path.join(package_dir, "configs", "setting.yaml")
    config_reader = ConfigReader(config_path)

    print(config_reader.get_checkpoint_config())
    print(config_reader.get_model_config())
    print(config_reader.get_loss_config())
    print(config_reader.get_optimizer_config())
    print(config_reader.get_list_metric_config())
