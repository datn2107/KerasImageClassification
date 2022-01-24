import os
import pandas as pd
import warnings
import tensorflow as tf


class DataLoader:
    def __init__(self, dataframe: pd.DataFrame, image_root: str,
                 batch_size: int = 8, height: int = 224, width: int = 224):
        """ Build tf.data.dataset from dataframe

        Dataframe need contain n+1 column where:
            Index column: is the column contain image name
            n other columns: are the label of image, value is 0 or 1
        All image WILL be resized to the same size
        But it WON'T be normalized in this class, it will be handled by preprocessing layer of model

        :param dataframe: Metadata of dataset you want to build
        :param image_root: Path to directory that contain image
        :param batch_size: Batch size
        :param height: Image size to resize
        :param width: Image size to resize
        """

        self.image_names = list(dataframe.index)
        self.image_paths = list(map(lambda image_name: os.path.join(image_root, image_name), self.image_names))
        self.labels = list(dataframe.apply(tuple, axis=1).values)
        self.batch_size = batch_size
        self.height, self.width = height, width

    def _load_image(self, path):
        image = tf.io.read_file(path)
        # We need to set expand_animations = False to make sure it return 3D tensor
        # because it can decode GIF image which return 4D tensor
        image = tf.io.decode_image(image, channels=3, expand_animations=False, dtype=tf.float32)
        if self.height is not None and self.width is not None:
            image = tf.image.resize(image, (self.height, self.width))
        else:
            image = tf.image.resize(image, (224, 224))

        return image

    def load_dataset(self, training=True):
        label_dataset = tf.data.Dataset.from_tensor_slices(self.labels)
        image_path_dataset = tf.data.Dataset.from_tensor_slices(self.image_paths)
        image_dataset = image_path_dataset.map(self._load_image, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
        if training:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


def load_dataset_from_root(data_root, batch_size, height=256, width=256):
    """ Data root must contain:
            - class_list.txt
            - train, val, test folder contain images
            - train_labels.csv, val_labels.csv, test_labels.csv are metadata of train, val, test
    """

    datasets = {}
    for phase in ['train', 'val', 'test']:
        image_root = os.path.join(data_root, phase)
        dataframe_path = os.path.join(data_root, phase + '_labels.csv')
        if os.path.exists(image_root) and os.path.exists(dataframe_path):
            dataframe = pd.read_csv(dataframe_path, index_col=0)
            datasets[phase] = DataLoader(dataframe, image_root, batch_size=batch_size,
                                         height=height, width=width).load_dataset(training=(phase == 'train'))
        else:
            datasets[phase] = None
            warnings.warn("No data for {} !!!".format(phase))

    return datasets

