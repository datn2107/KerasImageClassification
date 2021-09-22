import argparse
import os

import pandas as pd

from utils.config import ConfigReader
from utils.data import DataReader
from utils.evaluation import evaluate, save_result
from utils.model import KerasModel
from utils.compiler import compile_model

if __name__ == '__main__':
    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Config path')
    parser.set_defaults(config=os.path.join(package_dir, "configs", "setting.cfg"))

    config_reader = ConfigReader(parser.parse_args().config)
    path_info = config_reader.get_path()
    model_info = config_reader.get_model()
    data_info = config_reader.get_data()

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
    model = compile_model(model, optimizer_info=config_reader.get_optimizer(), loss_info=config_reader.get_loss(),
                          list_metric_info=config_reader.get_list_metric())

    if model_cp_dir is None and weights_cp_path is None:
        raise ValueError("There are no checkpoint to evaluate.")
    result = evaluate(model, test_dataset, model_cp_dir=model_cp_dir, weights_cp_path=weights_cp_path)
    print(result)

    save_result(result=result, saving_dir=saving_dir, model_name=model_info['model_name'])
