import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import argparse
from typing import List, Dict

import tensorflow as tf

from utils.config import ConfigReader


def load_optimizer(optimizer: str = "Adam", **parameters) -> tf.keras.optimizers:
    """
    :param optimizer: Name of the optimizer
    :param parameters:  Parameter for optimizer
    :return: tf.keras.optimizer
    """
    # didn't use .from_config of tf.keras.optimizers
    # Because it just allow some arguments suach as "clipnorm", "clipvalue", "lr", "decay", "global_clipnorm"
    optimizer = eval("tf.keras.optimizers." + optimizer)(**parameters)
    return optimizer


def load_loss(loss: str = "BinaryCrossentropy", **parameters) -> tf.keras.losses:
    """
    :param loss: Name of the loss
    :param parameters: Parameter for loss
    :return: tf.keras.losses
    """
    # didn't use .from_config of tf.keras.losses
    # Because it similar to optimizer is allow very few arguments
    loss = eval("tf.keras.losses." + loss)(**parameters)
    return loss


def load_metric(metric: str, **parameters) -> tf.keras.metrics.Metric:
    """
    :param loss: Name of the metric
    :param parameters: Parameter for metric
    :return: tf.keras.losses
    """
    metric = eval("tf.keras.metrics." + metric)(**parameters)
    return metric


def load_list_metric(list_metric_info: List[Dict]) -> List[tf.keras.metrics.Metric]:
    # User can define more than one metric in config so we need to create list of metric for that
    list_metric = ['accuracy']
    for metric_info in list_metric_info:
        list_metric.append(load_metric(**metric_info))
    return list_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Checkpoint Direction')
    parser.set_defaults(config=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../config/setting.cfg"))

    config_reader = ConfigReader(parser.parse_args().config)
    optimizer_info = config_reader.get_optimizer_config()
    loss_info = config_reader.get_loss_config()
    list_metric_info = config_reader.get_list_metric_config()

    print("\nOptimizer")
    print(load_optimizer(**optimizer_info).get_config())
    print()

    print("\nLoss")
    print(load_loss(**loss_info).get_config())
    print()

    list_metric = load_list_metric(list_metric_info)
    print("\nMetrics")
    for metric in list_metric:
        if metric != 'accuracy':
            print(metric.get_config())
    print()
