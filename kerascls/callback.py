import yaml
import os
import shutil

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard


def load_ckpt_callbacks(saving_root):
    ckpt_callbacks = []
    tf2_filepath = os.path.join(saving_root, 'tf2', 'epoch_{epoch:04d}')
    hdf5_filepath = os.path.join(saving_root, 'hdf5', 'epoch_{epoch:04d}.hdf5')

    for filepath in [tf2_filepath, hdf5_filepath]:
        ckpt_callbacks.append(ModelCheckpoint(filepath=filepath, verbose=1))

    return ckpt_callbacks


def load_best_ckpt_callbacks(saving_root, last_best_loss=None):
    best_ckpt_callbacks = []
    tf2_filepath = os.path.join(saving_root, 'tf2', 'best')
    hdf5_filepath = os.path.join(saving_root, 'hdf5', 'best.hdf5')

    for filepath in [tf2_filepath, hdf5_filepath]:
        best_ckpt_callbacks.append(ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True))
        if last_best_loss is not None:
            best_ckpt_callbacks[-1].best = last_best_loss

    return best_ckpt_callbacks


def save_config_to_saving_root(config_path, saving_root):
    config_file_name = os.path.basename(config_path)
    if config_file_name in os.listdir(saving_root):
        print('Config file is already contain in ' + saving_root)
        choice = input('Do you want to replace existed config: Y: Yes')
        if choice not in ['Y', 'y']:
            raise Exception('Cannot saving model.yaml file. Config file is already contain in ' + saving_root)

    try:
        shutil.copyfile(config_path, os.path.join(saving_root, config_file_name))
        print("Save config to Saving Directory: ", saving_root)
    except shutil.SameFileError:
        print("Already exist in ", saving_root)


class ChangingConfig(tf.keras.callbacks.Callback):
    """ Saving checkpoints and latest epoch to config file to resume training in next time

        It will modify 4 part in the Checkpoint section in config file:
            + weights_cp_root
            + weights_cp_path
            + best_weights_cp_path
            + last epcoh
    """

    def __init__(self, saving_root):
        super().__init__()
        with open(os.path.join(saving_root, "model.yaml"), 'r') as stream:
            self.config_data = yaml.safe_load(stream)
        self.saving_root = saving_root

    def _write_ckpt_2_config(self, epoch):
        tf2_ckpt_root = os.path.join(self.saving_root, "tf2")
        last_epoch_weight_root = os.path.join(tf2_ckpt_root, "epoch_{epoch:04d}".format(epoch=epoch+1), "variables")
        last_epoch_weight_path = os.path.join(last_epoch_weight_root, "variables")
        best_weight_path = os.path.join(tf2_ckpt_root, "best", "variables", "variables")

        self.config_data['Checkpoints']['weights_cp_root'] = last_epoch_weight_root
        self.config_data['Checkpoints']['weights_cp_path'] = last_epoch_weight_path
        self.config_data['Checkpoints']['best_weights_cp_path'] = best_weight_path

    def on_epoch_end(self, epoch, logs=None):
        self._write_ckpt_2_config(epoch)
        self.config_data['Checkpoints']['last_epoch'] = epoch + 1

        with open(os.path.join(self.saving_root, "model.yaml"), "w") as stream:
            yaml.dump(self.config_data, stream)


def load_callbacks(model_config_path, saving_root, last_best_loss=None, save_best_only=False):
    """ There all total 4 different call backs
            - Model checkpoint callbacks (tf2, h5 format)
                + Save model checkpoint in each epoch
                + Save best model checkpoint
            - Tensorboard callback
            - Csv Logger (display loss and metrics each epoch)
    """
    ckpt_callbacks = load_best_ckpt_callbacks(saving_root, last_best_loss=last_best_loss)
    if not save_best_only:
        ckpt_callbacks.extend(load_ckpt_callbacks(saving_root))

    tb_callback = TensorBoard(log_dir=os.path.join(saving_root, "tensor_board", "logs"),
                              histogram_freq=1,
                              profile_batch='500,520')
    csv_logger = CSVLogger(os.path.join(saving_root, "log.csv"), append=True)

    save_config_to_saving_root(model_config_path, saving_root)
    config_callback = ChangingConfig(saving_root)

    return [ckpt_callbacks, tb_callback, csv_logger, config_callback]
