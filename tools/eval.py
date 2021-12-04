import argparse
import os

import pandas as pd

from kerascls.config import ConfigReader
from kerascls.data import DataLoader
from tools.utils import load_and_compile_model_from_config, save_result

package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, help='Directory path contain evaluation image',
                    default=r'D:\Machine Learning Project\Fashion Recommend System')
parser.add_argument('--metadata_path', type=str, help='Dataframe contain metadata for evaluation data',
                    default=r'D:\Machine Learning Project\Fashion Recommend System')
parser.add_argument('--saving_dir', type=str, help='Directory path to save checkpoint of model and training result',
                    default=os.path.join(package_dir, "saving_dir"))
parser.add_argument('--config', type=str, help='Config path',
                    default=os.path.join(package_dir, "configs", "setting.yaml"))
parser_args = parser.parse_args()

# Check arguments
if not os.path.exists(parser_args.image_dir):
    raise ValueError('Image Directory is not exist')
if not os.path.exists(parser_args.metadata_path):
    raise ValueError('Metadata is not exist')
if not os.path.exists(parser_args.saving_dir):
    print('Create directory: ' + parser_args.saving_dir)
    os.makedirs(parser_args.saving_dir)

if __name__ == '__main__':
    # Load information from config
    config_reader = ConfigReader(parser.parse_args().config)
    model_info = config_reader.get_model_config()
    checkpoint = config_reader.get_checkpoint_config()

    saving_dir = parser_args.saving_dir
    test_dataframe = pd.read_csv(parser_args.metadata_path, index_col=0)

    # Load and Compile Model with Loss and Metric
    keras_model = load_and_compile_model_from_config(config_reader, len(test_dataframe.columns))

    # Load Dataset
    # input_shape of model [batch, height, width, channel]
    input_shape = keras_model.full_model.input_shape
    test_dataset = DataLoader(test_dataframe, parser_args.image_dir,
                              height=input_shape[1], width=input_shape[2]).load_dataset(training=False)

    # Evaluate Model
    keras_model.load_weights(weights_cp_path=checkpoint['best_weights_cp_path'])
    result = keras_model.full_model.evaluate(test_dataset, return_dict=True)
    print(result)

    # Save result
    save_result(result=result, saving_path=os.path.join(saving_dir, "evaluate_result"),
                model_name=model_info['model_name'])
