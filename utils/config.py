import configparser
from typing import List, Dict, Any


def _read_config(config_path: str):
    config_parser = configparser.ConfigParser(allow_no_value=True)
    config_parser.read(config_path)
    return config_parser

def _get_true_type_from_str(value: str) -> Any:
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

def _get_section(config_parser: configparser, section: str) -> Dict:
    section_value = dict(config_parser[section])
    for key, value in section_value.items():
        section_value[key] = get_true_type_from_str(value)
    return section_value

def get_path_from_config(config_path: str) -> Dict:
    path_config = get_section(read_config(config_path), 'Path')
    return path_config

def get_data_from_config(config_path: str) -> Dict:
    data_config = get_section(read_config(config_path), 'Data')
    if 'batch_size' not in data_config:
        data_config['batch_size'] = 8
    if 'epoch' not in data_config:
        data_config['epoch'] = 10
    if 'last_epoch' not in data_config:
        data_config['last_epoch'] = 0
    return data_config

def get_model_from_config(config_path: str) -> Dict:
    # Get config of base model
    model_config = get_section(read_config(config_path), 'Base Model')
    if 'model_name' not in model_config:
        model_config['model_name'] = 'ResNet50V2'
    # Get config of fully connected layer
    fully_connected_layer_config = get_section('Fully Connected Layer')
    if fully_connected_layer_config['num_dense'] <= 0:
        print("Number dense layer is less than 1. It will set to the default num_dense = 3")
        fully_connected_layer_config['num_dense'] = 3
    # combine two dictionary in one
    return {**model_config, **fully_connected_layer_config}

def get_optimizer_from_config(config_path: str) -> Dict:
    optimizer_config = get_section(read_config(config_path), 'Optimizer')
    return optimizer_config

def get_loss_from_config(config_path: str) -> Dict:
    loss_config = get_section(read_config(config_path), 'Loss')
    return loss_config

def get_list_metric_from_config(config_path: str) -> List[Dict]:
    list_metric_config = []

    config_parser = read_config(config_path)
    for section in config_parser.sections():
        if "Metric" not in section:
            continue
        metric_config = get_section(config_parser, section)
        list_metric_config.append(metric_config)
    return list_metric_config
