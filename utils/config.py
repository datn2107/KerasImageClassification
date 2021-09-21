import configparser
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

    def get_path_config(self) -> Dict:
        path_config = self.get_section('Path')
        return path_config

    def get_data_config(self) -> Dict:
        data_config = self.get_section('Data')
        if 'batch_size' not in data_config:
            data_config['batch_size'] = 8
        if 'epoch' not in data_config:
            data_config['epoch'] = 10
        if 'last_epoch' not in data_config:
            data_config['last_epoch'] = 0
        return data_config

    def get_model_config(self) -> Dict:
        # Get config of base model
        model_config = self.get_section('Base Model')
        if 'model_name' not in model_config:
            model_config['model_name'] = 'ResNet50V2'
        # Get config of fully connected layer
        fully_connected_layer_config = self.get_section('Fully Connected Layer')
        # combine two dictionary in one
        return {**model_config, **fully_connected_layer_config}

    def get_optimizer_config(self) -> Dict:
        optimizer_config = self.get_section('Optimizer')
        return optimizer_config

    def get_loss_config(self) -> Dict:
        loss_config = self.get_section('Loss')
        return loss_config

    def get_list_metric_config(self) -> List[Dict]:
        list_metric_config = []

        for section in self.config_parser.sections():
            if "Metric" not in section:
                continue
            metric_config = self.get_section(section)
            list_metric_config.append(metric_config)
        return list_metric_config
