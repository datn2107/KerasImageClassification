from typing import List, Dict

import tensorflow as tf


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


def load_metric(metric_name: str, **parameters) -> tf.keras.metrics.Metric:
    """
    :param metric_name: Name of the metric
    :param parameters: Parameter for metric
    :return: tf.keras.losses
    """
    metric = eval("tf.keras.metrics." + metric_name)(**parameters)
    return metric


def load_list_metric(list_metric_info: List[Dict]) -> List[tf.keras.metrics.Metric]:
    # User can define more than one metric in config so we need to create list of metric for that
    list_metric = ['accuracy']
    for metric_info in list_metric_info:
        list_metric.append(load_metric(**metric_info))
    return list_metric
