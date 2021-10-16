import tensorflow as tf


def exist_checkpoint(checkpoints):
    """Check whether checkpoint is was provide in config"""

    methods = ['weights_cp_path', 'weights_cp_dir']
    return any(checkpoints[val] is not None for val in methods)


def load_model(model_cp_dir=None, hdf5_cp_path=None, **ignore):
    """Load the full model with hdf5 file or directory that contain saved model"""
    # **ignore use to ignore undesired arguments while you passing by dict to function (**dict)

    if model_cp_dir is not None:
        model = tf.keras.models.load_model(model_cp_dir)
        print("Load checkpoints from ", model_cp_dir)
    elif hdf5_cp_path is not None:
        model = tf.keras.models.load_model(hdf5_cp_path)
        print("Load checkpoints from ", hdf5_cp_path)
    else:
        raise ValueError("There are no model checkpoint to load.")
    return model
