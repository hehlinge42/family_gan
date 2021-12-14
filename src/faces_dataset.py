import tensorflow as tf
import json

from typing import Text, Tuple, List
from logzero import logger
from sklearn.model_selection import train_test_split

FATHER = 0
MOTHER = 1


class FacesDataset:
    def __init__(self, summary_filepath):

        self.summary_filepath = summary_filepath
        self.dataset_train_len = None
        self.dataset_test_len = None

        self.seed = 42
        self.shuffle_buffer_size = 16
        self.num_parallel_calls = 8
        self.input_shape = (256, 256, 3)
        self.batch_size = 32

    @staticmethod
    def load_image(
        filepath: tf.Tensor, n_channels: int
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        image_file = tf.io.read_file(filename=filepath)
        image = tf.image.decode_png(
            contents=image_file, channels=n_channels, dtype=tf.uint8
        )
        return image

    def load_train_test(self) -> Tuple[List, List]:
        """
        Function to load train test data files from json

        Args:
            config_path (Text): Path to config file.

        Returns:
            Tuple[List, List]: Train and test image paths.

        """

        logger.info("Using presaved Train and Test data")

        with open(self.summary_filepath, "r") as fp:
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

    def normalize_y(self, X, y):

        y = tf.image.convert_image_dtype(y, dtype=tf.float32)
        y = tf.math.divide(y, 255.0)
        return X, y

    def get_train_test_dataset(self, is_train=True):
        def preprocess_dataset(X, y):

            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            for parent, child in dataset:
                logger.debug(f"parent: {type(parent)}, child: {type(child)}")
                break

            dataset = dataset.map(
                lambda x, y: (
                    (
                        FacesDataset.load_image(x[FATHER], 3),
                        FacesDataset.load_image(x[MOTHER], 3),
                    ),
                    FacesDataset.load_image(y, 3),
                ),
            )

            dataset = dataset.map(self.normalize_y)

            if is_train:
                dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
            else:
                dataset = dataset.batch(batch_size=1, drop_remainder=True)
            return dataset

        X_train, X_test, y_train, y_test = self.load_train_test()

        self.dataset_train_len = len(y_train)
        self.dataset_test_len = len(y_test)

        train_dataset = preprocess_dataset(X_train, y_train)
        test_dataset = preprocess_dataset(X_test, y_test)

        return train_dataset, test_dataset
