import tensorflow as tf
import json

from typing import Text, Tuple, List
from logzero import logger
from sklearn.model_selection import train_test_split

FATHER = 0
MOTHER = 1


class FacesDataset:
    def __init__(self):

        self.dataset_train_len = None
        self.dataset_test_len = None

        self.seed = 42
        self.shuffle_buffer_size = 16
        self.num_parallel_calls = 8
        self.input_shape = (256, 256, 3)
        self.batch_size = 32

    @staticmethod
    def load_image(
        filepaths: tf.Tensor, n_channels: int
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        image_file1 = tf.io.read_file(filename=filepaths[0])
        image1 = tf.image.decode_png(
            contents=image_file1, channels=n_channels, dtype=tf.uint8
        )
        image_file2 = tf.io.read_file(filename=filepaths[1])
        image2 = tf.image.decode_png(
            contents=image_file2, channels=n_channels, dtype=tf.uint8
        )
        image_file3 = tf.io.read_file(filename=filepaths[2])
        image3 = tf.image.decode_png(
            contents=image_file3, channels=n_channels, dtype=tf.uint8
        )
        return image1, image2, image3

    def load_train_test(self, dataset_summary_fp: Text) -> Tuple[List, List]:
        """
        Function to load train test data files from json

        Args:
            config_path (Text): Path to config file.

        Returns:
            Tuple[List, List]: Train and test image paths.

        """

        logger.info("Using presaved Train and Test data")

        with open(dataset_summary_fp, "r") as fp:
            dataset_summary = json.load(fp)
            X, y = [], []
            for idx, family in dataset_summary.items():
                if "son" in family:
                    X.append((family["father"], family["mother"]))
                    y.append(family["son"])
                if "daughter" in family:
                    X.append((family["father"], family["mother"]))
                    y.append(family["daughter"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.seed
        )

        return X_train, X_test, y_train, y_test

    def normalize_y(X, y):

        y = tf.image.convert_image_dtype(y, dtype=tf.float32)
        y = tf.math.divide(y, 255.0)
        return X, y

    def get_train_test_dataset(self, is_train=True):
        def preprocess_dataset(X, y):

            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

            dataset = dataset.map(
                lambda x, y: (
                    (
                        FacesDataset.load_image(x[FATHER]),
                        FacesDataset.load_image(x[MOTHER]),
                    ),
                    FacesDataset.load_image(y),
                ),
                self.num_parallel_calls,
            )

            dataset = dataset.map(self.normalize_y, self.num_parallel_calls)

            if is_train:
                dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
            else:
                dataset = dataset.batch(batch_size=1, drop_remainder=True)
            return dataset

        X_train, X_test, y_train, y_test = self.load_train_test(
            "../data/dataset_summary.json"
        )

        self.dataset_train_len = len(y_train)
        self.dataset_test_len = len(y_test)

        train_dataset = preprocess_dataset(X_train, y_train)
        test_dataset = preprocess_dataset(X_test, y_test)

        return train_dataset, test_dataset
