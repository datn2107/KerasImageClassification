import argparse
import os

import pandas as pd

from kerascls.callback import load_callbacks
from kerascls.checkpoint import load_model
from kerascls.config import ConfigReader
from kerascls.data import split_and_load_dataset
from kerascls.utils import load_and_compile_model_from_config, save_result, plot_log_csv

if __name__ == '__main__':
    # Get specific config path
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
    path_info = config_reader.get_path()
    model_info = config_reader.get_model()
    checkpoints = config_reader.get_checkpoint()

    saving_dir = path_info['saving_dir']
    dataframe = pd.read_csv(path_info['metadata_path'], index_col=0)

    # Load and Compile Model with Loss and Metric
    model = load_and_compile_model_from_config(config_reader, len(dataframe.columns))

    # Load Dataset
    # input_shape of model [batch, height, width, channel]
    input_shape = model.input_shape
    train_dataset, val_dataset, test_dataset = split_and_load_dataset(dataframe, path_info['image_dir'],
                                                                      batch_size=int(parser_args.batch),
                                                                      height=input_shape[1], width=input_shape[2])

    if any(val is not None for key, val in checkpoints.items() if key != "last_epoch"):
        loss_latest_checkpoint = model.evaluate(val_dataset, return_dict=True)['loss']
    else:
        loss_latest_checkpoint = None

    # Training
    model.fit(
        train_dataset,
        epochs=parser_args.epoch,
        validation_data=val_dataset,
        initial_epoch=checkpoints['last_epoch'],
        callbacks=load_callbacks(parser_args.config, saving_dir,
                                 loss_latest_checkpoint=loss_latest_checkpoint)
    )

    best_model_dir = os.path.join(saving_dir, "save_model", "best")
    model = load_model(model, model_cp_dir=best_model_dir)
    result = model.evaluate(test_dataset, return_dict=True)
    print(result)

    plot_log_csv(os.path.join(saving_dir, "log.csv"))
    save_result(result=result, saving_path=os.path.join(saving_dir, "evaluate_training_result"),
                model_name=model_info['model_name'])
