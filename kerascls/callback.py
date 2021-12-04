import configparser
import os
import shutil

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard


class ChangingConfig(tf.keras.callbacks.Callback):
    """Saving checkpoints and latest epoch to config file to resume training in next time"""

    def __init__(self, saving_dir):
        super().__init__()
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(saving_dir, "setting.yaml"))
        self.saving_dir = saving_dir

    def on_epoch_end(self, epoch, logs=None):
        section = 'Checkpoints'
        save_model_dir = os.path.join(self.saving_dir, "save_model")
        file_name = "epoch_{epoch:04d}".format(epoch=epoch + 1)
        model_cp_dir = os.path.join(save_model_dir, file_name)
        best_cp_dir = os.path.join(save_model_dir, 'best')
        # Save latest checkpoint path to config file
        self.config.set(section, 'weights_cp_dir', os.path.join(model_cp_dir, "variables"))
        self.config.set(section, 'weights_cp_path', os.path.join(model_cp_dir, "variables", "variables"))
        self.config.set(section, 'best_weights_cp_path', os.path.join(best_cp_dir, "variables", "variables"))
        # Save latest epoch to config file
        self.config.set(section, 'last_epoch', str(epoch + 1))
        with open(os.path.join(self.saving_dir, "setting.yaml"), "w") as configfile:
            self.config.write(configfile)


def load_callbacks(config_path, saving_dir, best_loss=None):
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

    # Load best loss to callback save best
    # if you resume training you need to load best loss that you already trained
    # if not the new best checkpoint is not include what you trained
    if best_loss is not None:
        # .best is save best loss
        save_best_model.best = best_loss
        save_best_model_hdf5.best = best_loss

    tb_callback = TensorBoard(log_dir=os.path.join(saving_dir, "tensor_board", "logs"),
                              histogram_freq=1,
                              profile_batch='500,520')
    csv_logger = CSVLogger(os.path.join(saving_dir, "log.csv"), append=True)

    # Save changing to config file to facility the resuming training
    if not os.path.exists(os.path.join(saving_dir, "setting.yaml")):
        shutil.copyfile(config_path, os.path.join(saving_dir, "setting.yaml"))
        print("Save config to Saving Directory: ", saving_dir)
    else:
        print('Config file is already contain in ' + saving_dir)
        choice = input('Do you want to replace existed config: Y: Yes')
        if choice not in ['Y','y']:
            raise Exception('Cannot saving setting.yaml file. Config file is already contain in ' + saving_dir)
        else:
            shutil.copyfile(config_path, os.path.join(saving_dir, "setting.yaml"))
            print("Save config to Saving Directory: ", saving_dir)
    save_config = ChangingConfig(saving_dir)

    return [save_model, save_best_model, save_model_hdf5, save_best_model_hdf5, tb_callback, csv_logger, save_config]
