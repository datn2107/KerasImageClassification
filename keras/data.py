import os

import tensorflow as tf
from sklearn.model_selection import train_test_split


class DataReader:
    def __init__(self, dataframe, image_dir, batch_size=8, height=None, width=None):
        self.list_image = list(dataframe.index)
        self.list_label = list(dataframe.apply(list, axis=1).values)
        self.list_image_path = list(map(lambda image: os.path.join(image_dir, image), self.list_image))
        self.batch_size = batch_size
        if height is None or width is None:
            self.height = self.width = 224
        else:
            self.height, self.width = height, width

    def _load_image(self, path):
        image = tf.io.read_file(path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.cast(image, tf.float32)
        # resize image depend on what model need
        # some model does need specific image size
        if self.height != None and self.width != None:
            image = tf.image.resize(image, (self.height, self.width))

        return image

    def load_dataset(self, training=True):
        label_tensor = list(map(tf.convert_to_tensor, self.list_label))
        label_dataset = tf.data.Dataset.from_tensor_slices(label_tensor)
        image_path_tensor = tf.data.Dataset.from_tensor_slices(self.list_image_path)
        image_dataset = image_path_tensor.map(self._load_image, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
        if training:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(self.batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


def split_and_load_dataset(dataframe, image_dir, batch_size, height, width, train_size=0.7):
    train_dataframe, test_dataframe = train_test_split(dataframe, train_size=train_size, shuffle=True,
                                                       random_state=2107)
    test_dataframe, val_dataframe = train_test_split(test_dataframe, train_size=0.5)

    train_dataset = DataReader(train_dataframe, image_dir, batch_size=batch_size, height=height, width=width).load_dataset(training=True)
    val_dataset = DataReader(val_dataframe, image_dir, batch_size, height, width).load_dataset(training=False)
    test_dataset = DataReader(test_dataframe, image_dir, batch_size, height, width).load_dataset(training=False)

    return train_dataset, val_dataset, test_dataset
