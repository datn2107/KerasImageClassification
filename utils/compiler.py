import warnings
from typing import List, Dict

import tensorflow as tf

from utils.loss_and_metric import load_optimizer, load_loss, load_list_metric


def compile_model(model: tf.keras.models.Model, optimizer_info: Dict, loss_info: Dict,
                  list_metric_info: List[Dict]) -> tf.keras.models.Model:
    model.compile(optimizer=load_optimizer(**optimizer_info),
                  loss=load_loss(**loss_info),
                  metrics=load_list_metric(list_metric_info))
    return model


def load_checkpoint(model, weights_cp_path=None, model_cp_dir=None):
    if model_cp_dir != None:
        model = tf.keras.models.load_model(model_cp_dir)
        print("Load checkpoint from ", model_cp_dir)
    elif weights_cp_path != None:
        model.load_weights(weights_cp_path)
        print("Load checkpoint from ", weights_cp_path, ".")
    else:
        warnings.warn("Does have any checkpoint to load.")
    return model
