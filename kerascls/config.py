import configparser
import os
import warnings
from typing import List, Dict, Any


class ConfigReader:
    def __init__(self, config_path: str):
        """Read data from config and return to dict for each section

        Allowed section:
            Path: Contain path to image directory, metadata and saving directory
            Data: Contain the fraction of each part of data (training, validation, testing)
            Checkpoint: Contain path to checkpoint
            Input Shape: Contain size of the image size to resize
            Base Model: Contain the argument of the base model
            Fully Connected Layer: Contain the config for fully connected layer
            Optimizer: Contain name of optimizer and arguments for its
            Loss: Contain name of loss and arguments for its
            Metric*: * is mean you can create more than 1 section with prefix string is "Metric"
                     Each section of this will contain name of metric and its arguments

        Using assert instead of raise to show raise error because assert will show specific error

        :param config_path: Absolute path to config file
        """

        self.config_parser = configparser.ConfigParser(allow_no_value=True)
        self.config_parser.read(config_path)

    @classmethod
    def _get_true_type_from_str(cls, value: str) -> Any:
        """Convert str to truth data type"""

        # Because all data read from config is str
        if value == "None":
            return None
        elif value == "True" or value == "False":
            return value == "True"
        elif value.isdigit():
            return int(value)
        else:
            try:
                return float(value)
            except ValueError:
                return value

    @classmethod
    def _fill_missing_key(cls, config_dict: Dict, allow_keys: List[str]) -> Dict:
        """Assign None to the missing arguments"""

        # Guarantee there are no missing arguments
        for key in allow_keys:
            if key not in config_dict:
                config_dict[key] = None
        return config_dict

    def get_section(self, section: str) -> Dict:
        """Get all information in section, convert to truth datatype and store it to dict"""

        section_value = dict(self.config_parser[section])
        for key, value in section_value.items():
            section_value[key] = self._get_true_type_from_str(value)
        return section_value

    def get_path_config(self) -> Dict:
        path_config = self.get_section('Path')
        assert path_config.get('image_dir', None) is None, "Missing directory path of image."
        assert path_config.get('metadata_path', None) is None, "Missing metadata path."

        # Set a default for saving directory
        # Default of saving direction is inside main package
        if path_config.get('saving_dir', None) is None:
            path_config['saving_dir'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saving_dir")
            warnings.warn(f"Saving directory is missing. It will save in %s" % path_config['saving_dir'])
        return path_config

    def get_data_config(self) -> Dict:
        data_config = self.get_section('Data')
        keys = ['train_size', 'val_size', 'test_size']
        assert any(data_config.get(key, None) is None for key in keys), 'Missing data size arguments'
        assert sum(data_config.values()) != 1., 'Sum of train, val and test data fraction is not equal 1'
        return data_config

    def get_checkpoint_config(self) -> Dict:
        checkpoint_config = self.get_section('Checkpoints')
        # Set None value instead of missing arguments
        allow_keys = ['model_cp_dir', 'hdf5_cp_path', 'weights_cp_path', 'weights_cp_dir']
        checkpoint_config = self._fill_missing_key(checkpoint_config, allow_keys)
        if 'last_epoch' not in checkpoint_config:
            checkpoint_config['last_epoch'] = 0
        return checkpoint_config

    def get_model_config(self) -> Dict:
        input_config = self.get_section('Input Shape')
        input_config = {'input_shape': (input_config.get('height', None), input_config.get('width', None), 3)}
        model_config = self.get_section('Base Model')
        fully_connected_layer_config = self.get_section('Fully Connected Layer')

        full_model_config = {**input_config, **model_config, **fully_connected_layer_config}

        # Only need to guarantee the model_name and num_dense are not missing or not None
        # because other arguments is has default value in build model class
        if full_model_config.get('model_name', None) is None:
            full_model_config['model_name'] = 'ResNet50V2'
        # guarantee that num_dense not less than 1
        if full_model_config.get('num_dense', None) is not None \
                and full_model_config.get('num_dense') <= 0:
            warnings.warn("Number dense layer is less than 1. It will set to the default num_dense = 1")
            full_model_config['num_dense'] = 1

        # Ignore parameter instead of None value in base_model and fully_connected_layer section
        # because this will pass to function by **dict
        # and parameters in function is already has default value
        # so if it has None value it will broke default value of function
        if full_model_config['input_shape'][0] is None or full_model_config['input_shape'][1] is None:
            full_model_config.pop('input_shape')
        keys = ['backbone_weights', 'trainable_backbone', 'last_pooling_layer',
                'num_dense', 'unit_first_dense_layer', 'units_remain_rate',
                'activation_dense', 'activation_last_dense', 'dropout_layer', 'dropout_rate']
        for key in keys:
            if key in full_model_config and full_model_config.get(key) is None:
                full_model_config.pop(key)

        return full_model_config

    def get_optimizer_config(self) -> Dict:
        optimizer_config = self.get_section('Optimizer')
        if 'optimizer' not in optimizer_config:
            optimizer_config['optimizer'] = 'SGD'
        return optimizer_config

    def get_loss_config(self) -> Dict:
        loss_config = self.get_section('Loss')
        if 'loss' not in loss_config:
            loss_config['loss'] = 'BinaryCrossentropy'
        return loss_config

    def get_list_metric_config(self) -> List[Dict]:
        list_metric_config = []

        for section in self.config_parser.sections():
            if "Metric" not in section:
                continue
            metric_config = self.get_section(section)
            if 'metric' not in metric_config:
                metric_config = 'BinaryAccuracy'
            list_metric_config.append(metric_config)
        return list_metric_config
