import argparse
import os
import numpy as np
import pandas as pd
import yaml

from kerascls.callback import load_callbacks
from tools.checkpoint import exist_checkpoint
from kerascls.config import ConfigReader
from kerascls.data import split_and_load_dataset, load_train_val_test
from tools.utils import load_and_compile_model_from_config, save_result, plot_log_csv

package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--data_config', type=str, help='Data config, which contain metadata and image path',
                    default=os.path.join(package_dir, "configs", "data.yaml"), required=True)
parser.add_argument('--saving_dir', type=str, help='Directory path to save checkpoint of model and training result',
                    default=os.path.join(package_dir, "saving_dir"))
parser.add_argument('--config', type=str, help='Config path',
                    default=os.path.join(package_dir, "configs", "setting.yaml"))
parser.add_argument('--num_class', type=int, help='Number Class', default=10, required=True)
parser.add_argument('--batch', type=int, help='Batch Size', default=32)
parser.add_argument('--epoch', type=int, help='Number Epoch', default=10)
parser_args = parser.parse_args()

# Check arguments
if not os.path.exists(parser_args.data_config):
    raise ValueError('Data config is not exist')
if not os.path.exists(parser_args.saving_dir):
    print('Create directory: ' + parser_args.saving_dir)
    os.makedirs(parser_args.saving_dir)


def load_data(data_config, input_shape):
    # input_shape: (batch, height, width, channel)

    with open(data_config, 'r') as stream:
        data_config = yaml.safe_load(stream)

    if data_config['splitted_data']['available']:
        splitted_date = data_config['splitted_data']
        dfs = []
        img_dirs = []
        for phase in ['train', 'val', 'test']:
            dfs.append(pd.read_csv(splitted_date['metadata_' + phase], index_col=0))
            img_dirs.append(splitted_date['img_' + phase])

        data_generator = load_train_val_test(dfs, img_dirs, batch_size=int(parser_args.batch),
                                             height=input_shape[1], width=input_shape[2], )
    elif data_config['raw_data']['available']:
        raw_data = data_config['raw_data']
        df = pd.read_csv(raw_data['metadata_path'], index_col=0)
        data_generator = split_and_load_dataset(df, raw_data['image_dir'],
                                                batch_size=int(parser_args.batch),
                                                height=input_shape[1], width=input_shape[2],
                                                train_size=raw_data['train_size'],
                                                val_size=raw_data['val_size'],
                                                test_size=raw_data['test_size'])
    else:
        raise ValueError("There are no data available in data_config")

    # return train_dataset, val_dataset, test_dataset
    return next(data_generator), next(data_generator), next(data_generator)


if __name__ == '__main__':
    # Load information from config
    config_reader = ConfigReader(parser_args.config)
    model_info = config_reader.get_model_config()
    checkpoints = config_reader.get_checkpoint_config()

    saving_dir = parser_args.saving_dir

    # Load and Compile Model with Loss and Metric
    keras_model = load_and_compile_model_from_config(config_reader, parser_args.num_class)

    # Load Dataset
    # Some model will need specific input shape to load weight or to have best performance
    # So the shape of input data will fit with input_shape of model
    input_shape = keras_model.full_model.input_shape  # (batch, height, width, channel)
    train_dataset, val_dataset, test_dataset = load_data(parser_args.data_config, input_shape)

    # Get best loss for resuming training
    if exist_checkpoint(checkpoints):
        keras_model.load_weights(weights_cp_path=checkpoints['best_weights_cp_path'])
        best_loss = keras_model.full_model.evaluate(val_dataset, return_dict=True)['loss']
    else:
        best_loss = None

    # Training
    keras_model.load_weights(**checkpoints)
    keras_model.full_model.fit(
        train_dataset,
        epochs=parser_args.epoch,
        validation_data=val_dataset,
        initial_epoch=checkpoints['last_epoch'],
        callbacks=load_callbacks(parser_args.config, saving_dir,
                                 best_loss=best_loss)
    )

    # Evaluate
    if test_dataset is not None:
        best_weights_cp_path = os.path.join(saving_dir, "save_model", "best", "variables", "variables")
        keras_model.load_weights(weights_cp_path=best_weights_cp_path)
        result = keras_model.full_model.evaluate(test_dataset, return_dict=True)
        print(result)
    else:
        result = np.inf

    plot_log_csv(os.path.join(saving_dir, "log.csv"))
    save_result(result=result, saving_path=os.path.join(saving_dir, "evaluate_training_result"),
                model_name=model_info['model_name'])
