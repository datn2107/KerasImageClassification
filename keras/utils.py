import json
import os
from typing import List, Dict

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from keras.loss_and_metric import load_optimizer, load_loss, load_list_metric


def compile_model(model: tf.keras.models.Model, optimizer_info: Dict, loss_info: Dict,
                  list_metric_info: List[Dict]) -> tf.keras.models.Model:
    model.compile(optimizer=load_optimizer(**optimizer_info),
                  loss=load_loss(**loss_info),
                  metrics=load_list_metric(list_metric_info))
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
    for index, train_label, val_label in enumerate(zip(train_column, val_column)):
        df.plot('epoch', [train_label, val_label])
        plt.savefig(os.path.join(log_dir, "log_{metric}.svg".format(metric=train_label)))


def save_result(result, saving_path, model_name):
    with open(saving_path, 'w') as f:
        f.write(json.dumps({model_name: result}))
