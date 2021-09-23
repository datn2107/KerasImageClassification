import configparser
import warnings
from typing import List, Dict, Any


class ConfigReader:
    def __init__(self, config_path: str):
        self.config_parser = configparser.ConfigParser(allow_no_value=True)
        self.config_parser.read(config_path)

    def get_true_type_from_str(cls, value: str) -> Any:
        if value == "None":
            return None
        elif value == "True" or value == "False":
            return (value == "True")
        else:
            try:
                return int(value)
            except Exception as e:
                try:
                    return float(value)
                except Exception as e:
                    pass
        return value

    def get_section(self, section: str) -> Dict:
        section_value = dict(self.config_parser[section])
        for key, value in section_value.items():
            section_value[key] = self.get_true_type_from_str(value)
        return section_value

    def get_path(self) -> Dict:
        path_config = self.get_section('Path')
        if path_config.get('image_dir', None) == None:
            raise ValueError("Missing directory path of image.")
        if path_config.get('metadata_path', None) == None:
            raise ValueError("Missing metadata path.")
        if path_config.get('saving_dir', None) == None:
            raise ValueError("Missing saving directory path to save model.")
        return path_config

    def get_checkpoint(self) -> Dict:
        checkpoint_config = self.get_section('Checkpoint')
        return checkpoint_config

    def get_data(self) -> Dict:
        data_config = self.get_section('Data')
        if 'batch_size' not in data_config:
            data_config['batch_size'] = 8
        if 'epoch' not in data_config:
            data_config['epoch'] = 10
        if 'last_epoch' not in data_config:
            data_config['last_epoch'] = 0
        return data_config

    def get_model(self) -> Dict:
        # Get config of base model
        model_config = self.get_section('Base Model')
        if 'model_name' not in model_config:
            model_config['model_name'] = 'ResNet50V2'
        # Get config of fully connected layer
        fully_connected_layer_config = self.get_section('Fully Connected Layer')
        if fully_connected_layer_config['num_dense'] <= 0:
            warnings.warn("Number dense layer is less than 1. It will set to the default num_dense = 3")
            fully_connected_layer_config['num_dense'] = 3
        # combine two dictionary in one
        return {**model_config, **fully_connected_layer_config}

    def get_optimizer(self) -> Dict:
        optimizer_config = self.get_section('Optimizer')
        return optimizer_config

    def get_loss(self) -> Dict:
        loss_config = self.get_section('Loss')
        return loss_config

    def get_list_metric(self) -> List[Dict]:
        list_metric_config = []

        for section in self.config_parser.sections():
            if "Metric" not in section:
                continue
            metric_config = self.get_section(section)
            list_metric_config.append(metric_config)
        return list_metric_config
