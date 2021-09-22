import configparser
import os
import shutil

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard


class SavingConfigCallback(tf.keras.callbacks.Callback):
    def __init__(self, saving_dir):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(saving_dir, "setting.cfg"))
        self.saving_dir = saving_dir

    def on_epoch_end(self, epoch, logs=None):
        self.config.set('Data', 'last_epoch', str(epoch+1))
        self.config.set('Path', 'model_cp_dir', str(epoch+1))
        with open(os.path.join(self.saving_dir, "setting.cfg"), "w") as configfile:
            self.config.write(configfile)


def load_callbacks(config_path, saving_dir, loss_lastest_checkpoint=None):
    file_name = "epoch_{epoch:04d}"

    save_model_dir = os.path.join(saving_dir, "save_model")
    save_model = ModelCheckpoint(filepath=os.path.join(save_model_dir, file_name), verbose=1)
    save_best_model = ModelCheckpoint(filepath=os.path.join(save_model_dir, "best"), verbose=1,
                                      save_best_only=True)

    save_model_h5_dir = os.path.join(saving_dir, "save_model", "hdf5")
    save_model_h5 = ModelCheckpoint(filepath=os.path.join(save_model_h5_dir, file_name + ".hdf5"), verbose=1)
    save_best_model_h5 = ModelCheckpoint(filepath=os.path.join(save_model_h5_dir, "best" + ".hdf5"),
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

    if config_path != os.path.join(saving_dir, "setting.cfg"):
        shutil.copyfile(config_path, os.path.join(saving_dir, "setting.cfg"))
    save_config = SavingConfigCallback(saving_dir)

    return [save_model, save_best_model, save_model_h5, save_best_model_h5, tb_callback, csv_logger, save_config]
