import os
import warnings

import tensorflow as tf


def load_latest_weight(weights_cp_dir=None):
    latest_checkpoint = None
    for checkpoint in os.listdir(weights_cp_dir):
        checkpoint = os.path.join(weights_cp_dir, checkpoint)
        if ".index" in checkpoint:
            if latest_checkpoint is None:
                latest_checkpoint = checkpoint
            elif os.path.getmtime(latest_checkpoint) < os.path.getmtime(checkpoint):
                latest_checkpoint = checkpoint
    return ".".join(latest_checkpoint.split('.')[:-1])


def load_checkpoint(model, model_cp_dir=None, hdf5_cp_path=None, weights_cp_dir=None, weights_cp_path=None, **ignore):
    if model_cp_dir is not None:
        model = tf.keras.models.load_model(model_cp_dir)
        print("Load checkpoints from ", model_cp_dir)
    elif hdf5_cp_path is not None:
        model = tf.keras.models.load_model(hdf5_cp_path)
        print("Load checkpoints from ", hdf5_cp_path)
    elif weights_cp_path is not None or weights_cp_dir is not None:
        if weights_cp_dir is not None:
            weights_cp_path = load_latest_weight(weights_cp_dir=weights_cp_dir)
        model = model.load_weights(weights_cp_path)
        print("Load checkpoints from ", weights_cp_path, ".")
    else:
        warnings.warn("Does have any checkpoints to load.")
    return model
