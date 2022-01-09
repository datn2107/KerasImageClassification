import warnings
from typing import List, Dict, Any

import yaml


class ModelConfigReader:
    """Read data from config and return to dict for each section.
        Warning: Config doesn't allow missing key, set it to None if you don't use it

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

    def get_checkpoint_config(self) -> Dict:
        """
        :return: Dict contain:
                - weights_cp_root (str): Path of director which contain weights checkpoints for model
                - weights_cp_path (str): Path of weights checkpoint for model
                - best_weights_cp_path (str): Path of best weights checkpoint for model
                - last_epoch (int): The epoch of the last training
        """
        checkpoint_config = self.config_data['Checkpoints']
        return checkpoint_config

    def _check_model_config(self, full_model_config):
        if full_model_config.get('model_name', None) is None:
            raise (ValueError, 'Missing model name in config')

        # guarantee that num_dense not less than 1
        if full_model_config.get('num_dense', 0) < 1:
            warnings.warn("Number dense layer is less than 1. It will set to the default num_dense = 1")
            full_model_config['num_dense'] = 1

        return full_model_config

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
                - remained_units_fraction (float): The fraction of remain unit
                - activation_dense (str): The activation of each dense layer
                - last_dense_activation (str): The activation of the last dense layer or the output layer
                - dropout_layer (bool): Add dropout after each dense layer or not
                - dropout_rate (float): Dropout rate of each dropout layer
        """

        def _check_model_config():
            if full_model_config.get('model_name', None) is None:
                raise (ValueError, 'Missing model name in config')

            # guarantee that num_dense not less than 1
            if full_model_config.get('num_dense', 0) < 1:
                warnings.warn("Number dense layer is less than 1. It will set to the default num_dense = 1")
                full_model_config['num_dense'] = 1

        def _eliminate_ono_value_key(config):
            # Eliminate parameter has None value
            # because it will eliminate the default value of function parameters
            processed_config = {key: value for key, value in config.items() if value is not None}
            return processed_config

        input_config = self.config_data['InputShape']
        model_config = self.config_data['BaseModel']
        fcl_config = self.config_data['FullyConnectedLayer']

        full_model_config = {**input_config, **model_config, **fcl_config}

        _check_model_config()
        full_model_config = _eliminate_ono_value_key(full_model_config)

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
    config_path = os.path.join(package_dir, "configs", "model.yaml")
    config_reader = ModelConfigReader(config_path)

    print(config_reader.get_checkpoint_config())
    print(config_reader.get_model_config())
    print(config_reader.get_loss_config())
    print(config_reader.get_optimizer_config())
    print(config_reader.get_list_metric_config())
