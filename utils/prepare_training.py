import os
import warnings
from typing import List, Dict

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard

from utils.prepare_compiler import load_optimizer, load_loss, load_list_metric


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


def load_callbacks(saving_dir, loss_lastest_checkpoint=None):
    file_name = "epoch_{epoch:04d}-val_loss_{val_loss:.2f}"

    save_model_dir = os.path.join(saving_dir, "save_model")
    save_model = ModelCheckpoint(filepath=os.path.join(save_model_dir, file_name), verbose=1)
    save_best_model = ModelCheckpoint(filepath=os.path.join(save_model_dir, "best_" + file_name), verbose=1,
                                      save_best_only=True)

    save_model_h5_dir = os.path.join(saving_dir, "save_model", "hdf5")
    save_model_h5 = ModelCheckpoint(filepath=os.path.join(save_model_h5_dir, file_name + ".hdf5"), verbose=1)
    save_best_model_h5 = ModelCheckpoint(filepath=os.path.join(save_model_h5_dir, "best_" + file_name + ".hdf5"),
                                         verbose=1,
                                         save_best_only=True)

    # Load check point of last epoch if resuming training
    if loss_lastest_checkpoint != None:
        # .best is save best loss
        save_best_model.best = loss_lastest_checkpoint
        save_best_model_h5.best = loss_lastest_checkpoint

    tb_callback = TensorBoard(log_dir=os.path.join(saving_dir, "tensor_board", "logs"),
                              histogram_freq=1,
                              profile_batch='500,520')
    csv_logger = CSVLogger(os.path.join(saving_dir, "log.csv"), append=True)

    return [save_model, save_best_model, save_model_h5, save_best_model_h5, tb_callback, csv_logger]
