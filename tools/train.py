import argparse
import os

import pandas as pd

from kerascls.callback import load_callbacks
from tools.checkpoint import exist_checkpoint
from kerascls.config import ConfigReader
from kerascls.data import split_and_load_dataset, load_train_val_test
from tools.utils import load_and_compile_model_from_config, save_result, plot_log_csv

package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parser = argparse.ArgumentParser()

###
parser.add_argument('--img_dir_train', type=str, help='Directory path contain image',
                    default=r'D:\Machine Learning Project\Fashion Recommend System')
parser.add_argument('--df_train', type=str, help='Dataframe contain metadata',
                    default=r'D:\Machine Learning Project\Fashion Recommend System')
parser.add_argument('--img_dir_val', type=str, help='Directory path contain image',
                    default=r'D:\Machine Learning Project\Fashion Recommend System')
parser.add_argument('--df_val', type=str, help='Dataframe contain metadata',
                    default=r'D:\Machine Learning Project\Fashion Recommend System')
parser.add_argument('--img_dir_test', type=str, help='Directory path contain image',
                    default=r'D:\Machine Learning Project\Fashion Recommend System')
parser.add_argument('--df_test', type=str, help='Dataframe contain metadata',
                    default=r'D:\Machine Learning Project\Fashion Recommend System')
###
parser.add_argument('--image_dir', type=str, help='Directory path contain image',
                    default=r'D:\Machine Learning Project\Fashion Recommend System')
parser.add_argument('--metadata_path', type=str, help='Dataframe contain metadata',
                    default=r'D:\Machine Learning Project\Fashion Recommend System')
parser.add_argument('--saving_dir', type=str, help='Directory path to save checkpoint of model and training result',
                    default=os.path.join(package_dir, "saving_dir"))
parser.add_argument('--config', type=str, help='Config path',
                    default=os.path.join(package_dir, "configs", "setting.cfg"))
parser.add_argument('--train_size', type=float, help='The fraction of training data',
                    default=0.7)
parser.add_argument('--val_size', type=float, help='The fraction of validation data',
                    default=0.15)
parser.add_argument('--test_size', type=float, help='The fraction of testing data',
                    default=0.15)
parser.add_argument('--batch', type=int, help='Batch Size',
                    default=32)
parser.add_argument('--epoch', type=int, help='Number Epoch',
                    default=10)

parser_args = parser.parse_args()

# Check arguments
# if not os.path.exists(parser_args.image_dir):
#     raise ValueError('Image Directory is not exist')
# if not os.path.exists(parser_args.metadata_path):
#     raise ValueError('Metadata is not exist')
if not os.path.exists(parser_args.saving_dir):
    print('Create directory: ' + parser_args.saving_dir)
    os.makedirs(parser_args.saving_dir)
# if parser_args.train_size + parser_args.val_size + parser_args.test_size != 1.:
#     raise ValueError('Sum of train, val and test data fraction is not equal 1')


if __name__ == '__main__':
    # Load information from config
    config_reader = ConfigReader(parser_args.config)
    model_info = config_reader.get_model_config()
    checkpoints = config_reader.get_checkpoint_config()

    saving_dir = parser_args.saving_dir
    dataframe = pd.read_csv(parser_args.df_train, index_col=0)

    # Load and Compile Model with Loss and Metric
    keras_model = load_and_compile_model_from_config(config_reader, len(dataframe.columns))

    # Load Dataset
    # Some model will need specific input shape to load weight or to have best performance
    # So the shape of input data will fit with input_shape of model
    input_shape = keras_model.full_model.input_shape  # (batch, height, width, channel)
    train_dataset, val_dataset, test_dataset = load_train_val_test(
        [parser_args.df_train, parser_args.df_val, parser_args.df_test],
        [parser_args.img_dir_train, parser_args.img_dir_val, parser_args.img_dir_test],
        batch_size=int(parser_args.batch), height=input_shape[1], width=input_shape[2],)
    # train_dataset, val_dataset, test_dataset = split_and_load_dataset(dataframe, parser_args.image_dir,
    #                                                                   batch_size=int(parser_args.batch),
    #                                                                   height=input_shape[1], width=input_shape[2],
    #                                                                   train_size=parser_args.train_size,
    #                                                                   val_size=parser_args.val_size,
    #                                                                   test_size=parser_args.test_size)

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
