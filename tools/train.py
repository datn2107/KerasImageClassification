import argparse
import os

import pandas as pd

from keras.callback import load_callbacks
from keras.checkpoint import load_checkpoint
from keras.config import ConfigReader
from keras.data import split_and_load_dataset
from keras.model import KerasModel
from keras.utils import compile_model, save_result, plot_log_csv

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

    # Load model and data
    if checkpoints['model_cp_dir'] is not None or checkpoints['hdf5_cp_path'] is not None:
        # Load full model from config
        model = load_checkpoint(None, **checkpoints)
    else:
        model_generator = KerasModel(**model_info, num_class=len(dataframe.columns))
        model = model_generator.create_model_keras()

    # Compile Model
    model = compile_model(model, optimizer_info=config_reader.get_optimizer(), loss_info=config_reader.get_loss(),
                          list_metric_info=config_reader.get_list_metric())

    # Load Dataset
    # input_shape of model [batch, height, width, channel]
    input_shape = model.input_shape
    train_dataset, val_dataset, test_dataset = split_and_load_dataset(dataframe, path_info['image_dir'],
                                                                      batch_size=int(parser_args.batch),
                                                                      height=input_shape[1], width=input_shape[2])

    model = load_checkpoint(model, **checkpoints)
    if any(checkpoint is not None for checkpoint in checkpoints.values()):
        loss_latest_epoch = model.evaluate(test_dataset, return_dict=True)['val_loss']
    else:
        loss_latest_epoch = None

    # Training
    history = model.fit(
        train_dataset,
        epochs=parser_args.epoch,
        validation_data=val_dataset,
        initial_epoch=checkpoints['last_epoch'],
        callbacks=load_callbacks(parser.parse_args().config, saving_dir,
                                 loss_lastest_checkpoint=loss_latest_epoch)
    )

    best_model_dir = os.path.join(saving_dir, "save_model", "best")
    model = load_checkpoint(model, model_cp_dir=best_model_dir)
    result = model.evaluate(test_dataset, return_dict=True)
    print(result)

    plot_log_csv(os.path.join(saving_dir, "log.csv"))
    save_result(result=result, saving_path=os.path.join(saving_dir, "evaluate_training_result"),
                model_name=model_info['model_name'])
