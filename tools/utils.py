import json
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from kerascls.config import ConfigReader
from kerascls.loss_and_metric import load_optimizer, load_loss, load_list_metric
from kerascls.model import KerasModel


def display_summary_model(model: tf.keras.models.Model):
    """Display Model"""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("---------------------------------Model---------------------------------")
        print(model.summary())
        print("")


def display_training_argumentation(config_reader: ConfigReader):
    """Display Optimizer, Loss and Metrics"""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("-----------------------------------------------Optimizer-----------------------------------------------")
        print(load_optimizer(**config_reader.get_optimizer_config()).get_config())
        print("")
        print("-------------------------------------------------Loss-------------------------------------------------")
        print(load_loss(**config_reader.get_loss_config()).get_config())
        print("")
        print("------------------------------------------------Metrics------------------------------------------------")
        metrics = load_list_metric(config_reader.get_list_metric_config())
        for metric in metrics:
            if metric != 'accuracy':
                print(metric.get_config())
        print("")


def load_and_compile_model_from_config(config_reader: ConfigReader, num_class: int = None) -> tf.keras.models.Model:
    """Load model can compile it with optimizer, loss and metrics from config_reader

    :param config_reader: ConfigReader that has information of model, loss, optimizer and metrics
    :param num_class: Number of classes to classify images
    """
    model_info = config_reader.get_model_config()

    # Load model and data
    # load full model from config
    keras_model = KerasModel(**model_info, num_class=num_class)
    keras_model.create_full_model()

    # Compile Model
    keras_model.compile(loss_info=config_reader.get_loss_config(),
                        optimizer_info=config_reader.get_optimizer_config(),
                        metrics_info=config_reader.get_list_metric_config())

    # Display
    display_summary_model(keras_model.full_model)
    display_training_argumentation(config_reader)

    return keras_model


def plot_log_csv(log_path):
    """Plot log csv file for visualization"""

    df = pd.read_csv(log_path)
    columns = df.columns

    # Get all metrics in log file
    # each column is the result of correspond metric
    # name of the column it also the name metric
    # each metric will have 2 column
    #   one has only metric name determine train result
    #   one has the prefix val_ determine validation result
    train_column = []
    val_column = []
    for column in columns:
        if "val" not in column and "epoch" != column:
            train_column.append(column)
            val_column.append("val_" + column)

    # Each Metric will plot into one graph
    log_dir = os.path.dirname(log_path)
    for index, (train_label, val_label) in enumerate(zip(train_column, val_column)):
        df.plot('epoch', [train_label, val_label])
        plt.savefig(os.path.join(log_dir, "log_{metric}.svg".format(metric=train_label)))


def save_result(result, saving_path, model_name):
    """Saving the result of evaluation on test dataset and with the best checkpoint"""

    # This also add a model name to the file
    with open(saving_path, 'w') as f:
        f.write(json.dumps({model_name: result}))
