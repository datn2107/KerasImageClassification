import argparse
import os

import pandas as pd

from utils.callback import load_callbacks
from utils.compiler import compile_model, load_checkpoint
from utils.config import ConfigReader
from utils.data import split_and_load_dataset
from utils.evaluation import evaluate, save_result, plot_log_csv
from utils.model import KerasModel

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
    dataframe = pd.read_csv(path_info['metadata_path'], index_col=0)

    # Load model and data
    model_generator = KerasModel(**model_info, num_class=len(dataframe.columns))
    model = model_generator.create_model_keras()

    # input_shape of model [batch, height, width, channel]
    input_shape = model.input_shape
    train_dataset, val_dataset, test_dataset = split_and_load_dataset(dataframe, path_info['image_dir'],
                                                                      batch_size=int(data_info['batch_size']),
                                                                      height=input_shape[1], width=input_shape[2])

    # Compile Model
    model = compile_model(model, optimizer_info=config_reader.get_optimizer(), loss_info=config_reader.get_loss(),
                          list_metric_info=config_reader.get_list_metric())
    model = load_checkpoint(model, model_cp_dir=model_cp_dir, weights_cp_path=weights_cp_path)
    if model_cp_dir is not None or weights_cp_path is not None:
        loss_lastest_checkpoint = evaluate(model, val_dataset)
    else:
        loss_lastest_checkpoint = None

    # Training
    history = model.fit(
        train_dataset,
        epochs=data_info['epoch'],
        validation_data=val_dataset,
        initial_epoch=data_info['last_epoch'],
        callbacks=load_callbacks(parser.parse_args().config, saving_dir,
                                 loss_lastest_checkpoint=loss_lastest_checkpoint)
    )

    best_model_path = os.path.join(saving_dir, "save_model", "best")
    result = evaluate(model, test_dataset, model_cp_dir=best_model_path)
    print(result)

    plot_log_csv(os.path.join(saving_dir, "log.csv"))
    save_result(result=result, saving_dir=saving_dir, model_name=model_info['model_name'])
