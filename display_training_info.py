import os
import argparse

from kerascls.loss_and_metric import load_loss, load_optimizer, load_list_metric
from kerascls.config import ModelConfigReader
from kerascls.model import KerasModelGenerator

PACKAGE_FILE = os.path.dirname(os.path.realpath(__file__))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, help='Model config path',
                        default=os.path.join(PACKAGE_FILE, "configs", "model.yaml"))
    parser.add_argument('--num_classes', type=int, help='Number classes', default=10)
    parser_args = parser.parse_args()

    return parser_args


def display_model(config_reader):
    model_config = config_reader.get_model_config()
    model_generator = KerasModelGenerator(**model_config)
    model_generator.create_model(num_classes=parser_args.num_classes)

    model = model_generator.full_model
    print(model.summary())


def display_loss(config_reader):
    print(load_loss(**config_reader.get_loss_config()).get_config())


def display_optimizer(config_reader):
    print(load_optimizer(**config_reader.get_optimizer_config()).get_config())


def display_metrics(config_reader):
    metrics = load_list_metric(config_reader.get_list_metric_config())
    for metric in metrics:
        if metric != 'accuracy':
            print(metric.get_config())
    print("")


if __name__ == '__main__':
    parser_args = parse_arguments()

    config_reader = ModelConfigReader(parser_args.model_config)

    display_model(config_reader)
    display_loss(config_reader)
    display_optimizer(config_reader)
    display_metrics(config_reader)
