import argparse
import os

import pandas as pd
from utils.config import *
from utils.data import DataReader
from utils.evaluation import evaluate, save_result
from utils.model import KerasModel
from utils.prepare_training import compile_model

if __name__ == '__main__':
    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config', type=str, help='Path to config contain reference')
    parser.set_defaults(path_config=os.path.join(package_dir, "configs", "path.cfg"))
    parser.add_argument('--model_config', type=str, help='Path of config contain model')
    parser.set_defaults(model_config=os.path.join(package_dir, "configs", "model.cfg"))
    parser.add_argument('--training_arg_config', type=str, help='Path of config contain training arguments')
    parser.set_defaults(training_arg_config=os.path.join(package_dir, "configs", "training.cfg"))

    path_info = get_path_from_config(parser.parse_args().path_config)
    model_info = get_model_from_config(parser.parse_args().model_config)
    data_info = get_data_from_config(parser.parse_args().training_arg_config)
    optimizer_info = get_optimizer_from_config(parser.parse_args().training_arg_config)
    loss_info = get_loss_from_config(parser.parse_args().training_arg_config)
    list_metric_info = get_list_metric_from_config(parser.parse_args().training_arg_config)

    saving_dir = path_info['saving_dir']
    model_cp_dir = path_info['model_cp_dir']
    weights_cp_path = path_info['weights_cp_path']
    test_dataframe = pd.read_csv(path_info['metadata_path'], index_col=0)

    # Load model and data
    model_generator = KerasModel(**model_info, num_class=len(test_dataframe.columns))
    model = model_generator.create_model_keras()

    # input_shape of model [batch, height, width, channel]
    input_shape = model.input_shape
    test_dataset = DataReader(test_dataframe, path_info['image_path'],
                              height=input_shape[1], width=input_shape[2]).load_dataset(training=False)

    # Compile Model
    model = compile_model(model, optimizer_info=optimizer_info, loss_info=loss_info, list_metric_info=list_metric_info)

    if model_cp_dir is not None or weights_cp_path is not None:
        raise ValueError("There are no checkpoint to evaluate.")
    result = evaluate(model, test_dataset, model_cp_dir=model_cp_dir, weights_cp_path=weights_cp_path)
    print(result)

    save_result(result=result, saving_dir=saving_dir, model_name=model_info['model_name'])
