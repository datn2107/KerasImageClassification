import argparse
import os

import pandas as pd
import tensorflow as tf

from keras_classification.utils.config import ConfigReader
from keras_classification.utils.data import split_and_load_dataset
from keras_classification.utils.evaluation import evaluate, save_result, plot_log_csv
from keras_classification.utils.model import KerasModel
from keras_classification.utils.prepare_training import compile_model, load_checkpoint, load_callbacks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Checkpoint Direction')
    parser.set_defaults(config=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../setting.cfg"))

    config_reader = ConfigReader(parser.parse_args().config)
    path_infor = config_reader.get_path_config()
    data_infor = config_reader.get_data_config()
    model_infor = config_reader.get_model_config()

    saving_dir = path_infor['saving_dir']
    dataframe = pd.read_csv(path_infor['metadata_path'], index_col=0)

    model_generator = KerasModel(**model_infor, num_class=len(dataframe.columns))
    model = compile_model(model_generator, config_reader)

    height, width, channel = model_generator.get_model_input_shape()
    _, _, test_dataset = split_and_load_dataset(dataframe, path_infor['image_path'],
                                                batch_size=int(data_infor['batch_size']),
                                                height=height, width=width)

    model = load_checkpoint(model, **path_infor)
    result = model.evaluate(test_dataset)
    print(result)

    save_result(result=result, saving_dir=saving_dir, model_name=model_infor['model_name'])
