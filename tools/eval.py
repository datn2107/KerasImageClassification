import argparse
import os

import pandas as pd

from kerascls.config import ConfigReader
from kerascls.data import DataReader
from kerascls.utils import load_and_compile_model_from_config, save_result

if __name__ == '__main__':
    # Get specific config path
    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Config path')
    parser.set_defaults(config=os.path.join(package_dir, "configs", "setting.cfg"))

    # Load information from config
    config_reader = ConfigReader(parser.parse_args().config)
    path_info = config_reader.get_path()
    model_info = config_reader.get_model()
    checkpoints = config_reader.get_checkpoint()
    if any(val is not None for key, val in checkpoints.items() if key != "last_epoch"):
        raise ValueError("There are no checkpoints to evaluate.")

    saving_dir = path_info['saving_dir']
    test_dataframe = pd.read_csv(path_info['metadata_path'], index_col=0)

    # Load and Compile Model with Loss and Metric
    model = load_and_compile_model_from_config(config_reader, len(test_dataframe.columns))

    # Load Dataset
    # input_shape of model [batch, height, width, channel]
    input_shape = model.input_shape
    test_dataset = DataReader(test_dataframe, path_info['image_dir'],
                              height=input_shape[1], width=input_shape[2]).load_dataset(training=False)

    # Evaluate Model
    result = model.evaluate(test_dataset, return_dict=True)
    print(result)

    # Save result
    save_result(result=result, saving_path=os.path.join(saving_dir, "evaluate_result"),
                model_name=model_info['model_name'])
