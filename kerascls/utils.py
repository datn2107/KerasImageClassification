import json
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from kerascls.checkpoint import load_model, load_weight
from kerascls.config import ConfigReader
from kerascls.loss_and_metric import load_optimizer, load_loss, load_list_metric
from kerascls.model import KerasModel


def display_summary(model: tf.keras.models.Model, config_reader: ConfigReader):
    # Display Model, Optimizer, Loss and Metrics
    print("---------------------------------Model---------------------------------")
    print(model.summary())
    print("-------------------------------Optimizer-------------------------------")
    print(load_optimizer(**config_reader.get_optimizer()).get_config())
    print("---------------------------------Loss---------------------------------")
    print(load_loss(**config_reader.get_loss()))
    print("--------------------------------Metrics--------------------------------")
    metrics = load_list_metric(config_reader.get_list_metric())
    for metric in metrics:
        if metric != 'accuracy':
            print(metric.get_config())


def load_and_compile_model_from_config(config_reader: ConfigReader, num_class: int = None) -> tf.keras.models.Model:
    model_info = config_reader.get_model()
    checkpoints = config_reader.get_checkpoint()

    # Load model and data
    # load full model from config
    model = load_model(None, **checkpoints)
    if model is None:
        model_generator = KerasModel(**model_info, num_class=num_class)
        model = model_generator.create_model_keras()
        model = load_weight(model, **checkpoints)

    # Compile Model
    model.compile(optimizer=load_optimizer(**config_reader.get_optimizer()),
                  loss=load_loss(**config_reader.get_loss()),
                  metrics=load_list_metric(config_reader.get_list_metric()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        display_summary(model, config_reader)

    return model


def plot_log_csv(log_path):
    df = pd.read_csv(log_path)
    columns = df.columns

    train_column = []
    val_column = []
    for column in columns:
        if "val" not in column and "epoch" != column:
            train_column.append(column)
            val_column.append("val_" + column)

    log_dir = os.path.dirname(log_path)
    for index, (train_label, val_label) in enumerate(zip(train_column, val_column)):
        df.plot('epoch', [train_label, val_label])
        plt.savefig(os.path.join(log_dir, "log_{metric}.svg".format(metric=train_label)))


def save_result(result, saving_path, model_name):
    with open(saving_path, 'w') as f:
        f.write(json.dumps({model_name: result}))
