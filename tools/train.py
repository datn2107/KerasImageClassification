import argparse
import os

import pandas as pd

from kerascls.callback import load_callbacks
from tools.checkpoint import exist_checkpoint
from kerascls.config import ConfigReader
from kerascls.data import split_and_load_dataset
from tools.utils import load_and_compile_model_from_config, save_result, plot_log_csv

if __name__ == '__main__':
    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Config path')
    parser.set_defaults(config=os.path.join(package_dir, "configs", "setting.cfg"))

    parser.add_argument('--batch', type=int, help='Batch Size')
    parser.set_defaults(batch=32)
    parser.add_argument('--epoch', type=int, help='Number Epoch')
    parser.set_defaults(epoch=10)

    parser_args = parser.parse_args()

    # Load information from config
    config_reader = ConfigReader(parser_args.config)
    path_info = config_reader.get_path_config()
    data = config_reader.get_data_config()
    model_info = config_reader.get_model_config()
    checkpoints = config_reader.get_checkpoint_config()

    saving_dir = path_info['saving_dir']
    dataframe = pd.read_csv(path_info['metadata_path'], index_col=0)

    # Load and Compile Model with Loss and Metric
    keras_model = load_and_compile_model_from_config(config_reader, len(dataframe.columns))

    # Load Dataset
    # Some model will need specific input shape to load weight or to have best performance
    # So the shape of input data will fit with input_shape of model
    input_shape = keras_model.full_model.input_shape  # (batch, height, width, channel)
    train_dataset, val_dataset, test_dataset = split_and_load_dataset(dataframe, path_info['image_dir'],
                                                                      batch_size=int(parser_args.batch),
                                                                      height=input_shape[1], width=input_shape[2],
                                                                      train_size=data['train_size'],
                                                                      val_size=data['val_size'],
                                                                      test_size=data['test_size'])

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
    best_weights_cp_path = os.path.join(saving_dir, "save_model", "best", "variables", "variables")
    keras_model.load_weights(weights_cp_path=best_weights_cp_path)
    result = keras_model.full_model.evaluate(test_dataset, return_dict=True)
    print(result)

    plot_log_csv(os.path.join(saving_dir, "log.csv"))
    save_result(result=result, saving_path=os.path.join(saving_dir, "evaluate_training_result"),
                model_name=model_info['model_name'])
