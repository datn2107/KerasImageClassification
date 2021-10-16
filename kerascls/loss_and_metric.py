from typing import List, Dict

import tensorflow as tf

def load_optimizer(optimizer: str, **arguments) -> tf.keras.optimizers.Optimizer:
    """ Build the optimizer from desired arguments.

    Optimizer is using from tf.keras.optimizers module.
    Arguments pass to this function need to be appropriate to selected optimizer.
    You can chose optimizer and see allowed arguments of selected optimizer here.
    [https://www.tensorflow.org/api_docs/python/tf/keras/optimizers]

    :param optimizer: Name of the optimizer
    :param arguments: Arguments for selected optimizer
    """

    # We didn't use .from_config of tf.keras.optimizers
    # Because it just allow some arguments such as "clipnorm", "clipvalue", "lr", "decay", "global_clipnorm"
    optimizer = eval("tf.keras.optimizers." + optimizer)(**arguments)
    return optimizer


def load_loss(loss: str, **arguments) -> tf.keras.losses.Loss:
    """ Build the loss from desired arguments.

    Loss is using from tf.keras.losses module.
    Arguments pass to this function need to be appropriate to selected loss.
    You can chose loss and see allowed arguments of selected loss here.
    [https://www.tensorflow.org/api_docs/python/tf/keras/losses]

    :param loss: Name of the loss
    :param arguments: Arguments for selected loss
    """

    # We didn't use .from_config of tf.keras.losses
    # Because it similar to optimizer is allow very few arguments
    loss = eval("tf.keras.losses." + loss)(**arguments)
    return loss


def load_metric(metric: str, **arguments) -> tf.keras.metrics.Metric:
    """ Build the metric from desired arguments.

    Metric is using from tf.keras.metrics module.
    Arguments pass to this function need to be appropriate to selected metric.
    You can chose metric and see allowed arguments of selected metric here.
    [https://www.tensorflow.org/api_docs/python/tf/keras/metrics]

    :param metric: Name of the metric
    :param arguments: Arguments for selected metric
    """

    metric = eval("tf.keras.metrics." + metric)(**arguments)
    return metric


def load_list_metric(list_metric_info: List[Dict]) -> List[tf.keras.metrics.Metric]:
    """ Load list of metrics.

    :param list_metric_info: List of dict contains metric name and arguments of it.
    """

    list_metric = ['accuracy']
    for metric_info in list_metric_info:
        list_metric.append(load_metric(**metric_info))
    return list_metric
