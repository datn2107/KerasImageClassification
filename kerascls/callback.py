import configparser
import os
import shutil

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard


class ChangingConfig(tf.keras.callbacks.Callback):
    """Saving checkpoints and latest epoch to config file to continue training"""  #
    def __init__(self, saving_dir):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(saving_dir, "setting.cfg"))
        self.saving_dir = saving_dir

    def on_epoch_end(self, epoch):
        save_model_dir = os.path.join(self.saving_dir, "save_model")
        model_cp_dir = os.path.join(save_model_dir, "epoch_{epoch:04d}".format(epoch=epoch + 1))
        hdf5_cp_path = os.path.join(save_model_dir, "hdf5", "epoch_{epoch:04d}.hdf5".format(epoch=epoch + 1))
        weight_cp_dir = os.path.join(save_model_dir, "variables")
        weight_cp_path = os.path.join(model_cp_dir, "variables", "variables")
        self.config.set('Checkpoint', 'model_cp_dir', model_cp_dir)
        self.config.set('Checkpoint', 'hdf5_cp_path', hdf5_cp_path)
        self.config.set('Checkpoint', 'weight_cp_dir', weight_cp_dir)
        self.config.set('Checkpoint', 'weight_cp_path', weight_cp_path)
        self.config.set('Data', 'last_epoch', str(epoch + 1))
        with open(os.path.join(self.saving_dir, "setting.cfg"), "w") as configfile:
            self.config.write(configfile)


def load_callbacks(config_path, saving_dir, loss_latest_checkpoint=None):
    # Save checkpoints
    file_name = "epoch_{epoch:04d}"

    save_model_dir = os.path.join(saving_dir, "save_model")
    save_model = ModelCheckpoint(filepath=os.path.join(save_model_dir, file_name), verbose=1)
    save_best_model = ModelCheckpoint(filepath=os.path.join(save_model_dir, "best"), verbose=1,
                                      save_best_only=True)

    save_model_hdf5_dir = os.path.join(saving_dir, "save_model", "hdf5")
    save_model_hdf5 = ModelCheckpoint(filepath=os.path.join(save_model_hdf5_dir, file_name + ".hdf5"), verbose=1)
    save_best_model_hdf5 = ModelCheckpoint(filepath=os.path.join(save_model_hdf5_dir, "best" + ".hdf5"),
                                           verbose=1,
                                           save_best_only=True)

    # load check point of last epoch if resuming training
    if loss_latest_checkpoint is not None:
        # .best is save best loss
        save_best_model.best = loss_latest_checkpoint
        save_best_model_hdf5.best = loss_latest_checkpoint

    #
    tb_callback = TensorBoard(log_dir=os.path.join(saving_dir, "tensor_board", "logs"),
                              histogram_freq=1,
                              profile_batch='500,520')
    csv_logger = CSVLogger(os.path.join(saving_dir, "log.csv"), append=True)

    # Save changing to config
    if config_path != os.path.join(saving_dir, "setting.cfg"):
        shutil.copyfile(config_path, os.path.join(saving_dir, "setting.cfg"))
    save_config = ChangingConfig(saving_dir)

    return [save_model, save_best_model, save_model_hdf5, save_best_model_hdf5, tb_callback, csv_logger, save_config]
