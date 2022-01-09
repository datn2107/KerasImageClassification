import argparse
import os
import json
import pandas as pd
import tensorflow as tf

from kerascls.model import KerasModelGenerator
from kerascls.config import ModelConfigReader
from kerascls.data import load_dataset_from_root
from kerascls.callback import load_callbacks

PACKAGE_FILE = os.path.dirname(os.path.realpath(__file__))


def check_parser_arguments(parser_args):
    if not os.path.exists(parser_args.data_root):
        raise ValueError('Data directory is not exist')
    if not os.path.exists(parser_args.saving_root):
        print('Create directory: ' + parser_args.saving_root)
        os.makedirs(parser_args.saving_root)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, help='Directory contain image and metadata',
                        default="../data")
    parser.add_argument('--saving_root', type=str,
                        help='Directory path to save checkpoint of model and training result',
                        default=os.path.join(PACKAGE_FILE, "saving_root"))
    parser.add_argument('--model_config', type=str, help='Model config path',
                        default=os.path.join(PACKAGE_FILE, "configs", "model.yaml"))
    parser.add_argument('--batch_size', type=int, help='Batch Size', default=2)
    parser.add_argument('--epoch', type=int, help='Number Epoch', default=10)
    parser.add_argument('--save_best_only', action='store_true')
    parser_args = parser.parse_args()

    return parser_args


def get_num_class(data_root):
    for phase in ['train', 'val', 'test']:
        image_root = os.path.join(data_root, phase)
        dataframe_path = os.path.join(data_root, phase + '_labels.csv')
        if os.path.exists(image_root) and os.path.exists(dataframe_path):
            dataframe = pd.read_csv(dataframe_path, index_col=0)
            return len(dataframe.columns)


def get_last_epoch(model_config_path):
    config_reader = ModelConfigReader(model_config_path)
    checkpoint_config = config_reader.get_checkpoint_config()

    if checkpoint_config['last_epoch'] is not None:
        return checkpoint_config['last_epoch']
    else:
        return 0


def prepare_model_generator(model_config_path, num_classes):
    def load_checkpoint():
        checkpoint_config = config_reader.get_checkpoint_config()
        model_generator.load_weights(**checkpoint_config)

    def compile_model():
        loss_config = config_reader.get_loss_config()
        optimizer_config = config_reader.get_optimizer_config()
        metrics_config = config_reader.get_list_metric_config()

        model_generator.compile_model(loss_config=loss_config,
                                      optimizer_config=optimizer_config,
                                      metrics_config=metrics_config)

    config_reader = ModelConfigReader(model_config_path)
    model_config = config_reader.get_model_config()
    model_generator = KerasModelGenerator(**model_config)

    model_generator.create_model(num_classes=num_classes)
    load_checkpoint()
    compile_model()

    return model_generator


if __name__ == '__main__':
    parser_args = parse_arguments()
    check_parser_arguments(parser_args)
    saving_root = parser_args.saving_root
    model_config_path = parser_args.model_config

    model_generator = prepare_model_generator(model_config_path,
                                              num_classes=get_num_class(parser_args.data_root))
    model = model_generator.full_model

    datasets = load_dataset_from_root(parser_args.data_root, batch_size=parser_args.batch_size,
                                      height=model_generator.input_shape[0], width=model_generator.input_shape[1])
    last_loss = model.evaluate(datasets['val'], batch_size=parser_args.batch_size,
                               return_dict=True)['loss']

    model.fit(
        datasets['train'],
        epochs=parser_args.epoch,
        validation_data=datasets['val'],
        initial_epoch=get_last_epoch(model_config_path),
        callbacks=load_callbacks(model_config_path=model_config_path,
                                 saving_root=saving_root,
                                 last_best_loss=last_loss,
                                 save_best_only=parser_args.save_best_only)
    )

    if datasets['test'] is not None:
        best_model_ckpt = os.path.join(saving_root, 'tf2', 'best')
        model = tf.keras.models.load_model(best_model_ckpt)
        result = model.evaluate(datasets['test'], return_dict=True)
        print(result)

        result_path = os.path.join(saving_root, "result")
        with open(result_path, 'w') as f:
            f.write(json.dumps(result))
