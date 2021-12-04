import os

import tensorflow as tf
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, dataframe, image_dir, batch_size=8, height=224, width=224):
        """ Build tf.data.dataset from dataframe

        Dataframe need contain n+1 column where:
            Index column is the column contain image name
            n other columns are the label of image, value is 0 or 1
        All image WILL be resize to the same size
        But it WON'T be normalize, it will be handle by preprocessing layer of model

        :param dataframe: Metadata of dataset you want to build
        :param image_dir: Path to directory that contain image
        :param batch_size: Batch size
        :param height: Image size to resize
        :param width: Image size to resize
        """

        self.list_image = list(dataframe.index)
        self.list_image_path = list(map(lambda image: os.path.join(image_dir, image), self.list_image))
        self.list_label = list(dataframe.apply(tuple, axis=1).values)
        self.batch_size = batch_size
        self.height, self.width = height, width

    def _load_image(self, path):
        image = tf.io.read_file(path)
        # We need to set expand_animations = False to make sure it return 3D tensor
        # because it can decode GIF image which return 4D tensor
        image = tf.io.decode_image(image, channels=3, expand_animations=False, dtype=tf.float32)
        # The model requires all image in the dataset to be the same size
        if self.height is not None and self.width is not None:
            image = tf.image.resize(image, (self.height, self.width))
        else:
            image = tf.image.resize(image, (224, 224))

        return image

    def load_dataset(self, training=True):
        label_dataset = tf.data.Dataset.from_tensor_slices(self.list_label)
        image_path_tensor = tf.data.Dataset.from_tensor_slices(self.list_image_path)
        image_dataset = image_path_tensor.map(self._load_image, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
        if training:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


def split_and_load_dataset(dataframe, image_dir, batch_size, height, width,
                           train_size=0.7, val_size=0.15, test_size=0.15):
    """Split and load data for training and testing"""

    # Split dataframe into 3 part training, validation and testing
    train_dataframe, test_dataframe = train_test_split(dataframe, train_size=train_size, shuffle=True,
                                                       random_state=2107)
    yield DataLoader(train_dataframe, image_dir, batch_size=batch_size, height=height,
                     width=width).load_dataset(training=True)

    val_dataframe = None
    if test_size != 0 and val_size != 0:
        test_dataframe, val_dataframe = train_test_split(test_dataframe, train_size=test_size / (val_size + train_size))
    elif val_size != 0:
        val_dataframe = test_dataframe

    yield (DataLoader(val_dataframe, image_dir, batch_size, height, width).load_dataset(training=False)
           if val_size != 0 else None)

    yield (DataLoader(test_dataframe, image_dir, batch_size, height, width).load_dataset(training=False)
           if test_size != 0 else None)


def load_train_val_test(dataframe, image_dir, batch_size, height, width):
    # Load dataset for each part
    # i == 0: train_dataset, i == 1: val_dataset, i == 2: test_dataset
    for i in range(len(dataframe)):
        yield (DataLoader(dataframe[i], image_dir[i], batch_size=batch_size, height=height, width=width).load_dataset(
            training=(i == 0)) if dataframe[i] is not None else None)
