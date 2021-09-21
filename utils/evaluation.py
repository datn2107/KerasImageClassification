import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.prepare_training import load_checkpoint


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


def save_result(result, saving_dir, model_name):
    with open(os.path.join(saving_dir, "result.json"), 'w') as f:
        f.write(json.dumps({model_name: result}))


def evaluate(model, test_dataset, weights_cp_path=None, model_cp_dir=None):
    if model_cp_dir == None and weights_cp_path == None:
        print("There are no additional checkpoint !!!")
    else:
        model = load_checkpoint(model, weights_cp_path=weights_cp_path, model_cp_dir=model_cp_dir)

    result = model.evaluate(test_dataset, return_dict=True)
    print(result)
