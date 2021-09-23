import argparse
import os

import pandas as pd

from kerascls.checkpoint import load_checkpoint
from kerascls.config import ConfigReader
from kerascls.data import DataReader
from kerascls.model import KerasModel
from kerascls.utils import compile_model, save_result

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

    saving_dir = path_info['saving_dir']
    test_dataframe = pd.read_csv(path_info['metadata_path'], index_col=0)

    # Load model and data
    if checkpoints['model_cp_dir'] is not None or checkpoints['hdf5_cp_path'] is not None:
        # Load full model from config
        model = load_checkpoint(None, **checkpoints)
    else:
        model_generator = KerasModel(**model_info, num_class=len(test_dataframe.columns))
        model = model_generator.create_model_keras()

    # Compile Model
    model = compile_model(model, optimizer_info=config_reader.get_optimizer(), loss_info=config_reader.get_loss(),
                          list_metric_info=config_reader.get_list_metric())

    # Load Dataset
    # input_shape of model [batch, height, width, channel]
    input_shape = model.input_shape
    test_dataset = DataReader(test_dataframe, path_info['image_dir'],
                              height=input_shape[1], width=input_shape[2]).load_dataset(training=False)

    # Evaluate Model
    if any(checkpoint is not None for checkpoint in checkpoints.values()):
        raise ValueError("There are no checkpoints to evaluate.")
    model = load_checkpoint(model, **checkpoints)
    result = model.evaluate(test_dataset, return_dict=True)
    print(result)

    # Save result
    save_result(result=result, saving_path=os.path.join(saving_dir, "evaluate_result"),
                model_name=model_info['model_name'])
