from faces_dataset import FacesDataset

import tensorflow_hub as hub
import tensorflow as tf
import os

from tensorflow.keras.layers import Flatten
from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError


def get_model():
    def conv(efficientnet, conv2d, bn, inp):
        x = efficientnet(inp, True)
        x = conv2d(x)
        x = bn(x)
        x = Flatten()(x)
        return x

    inp1_full_size = keras.layers.Input(shape=(256, 256, 3))
    inp2_full_size = keras.layers.Input(shape=(256, 256, 3))

    inp1 = tf.image.resize(inp1_full_size, (128, 128))
    inp2 = tf.image.resize(inp2_full_size, (128, 128))

    efficientnet = tf.keras.applications.efficientnet.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(128, 128, 3),
        input_tensor=None,
        pooling=None,
    )
    # efficientnet = tf.keras.applications.efficientnet.EfficientNetB5(
    #     include_top=False, weights='imagenet', input_shape=(128, 128, 3), input_tensor=None, pooling=None,
    # )

    conv2d = keras.layers.Conv2D(
        filters=512, kernel_size=(4, 4), bias_initializer="zeros"
    )
    bn = keras.layers.BatchNormalization()

    parent1 = conv(efficientnet, conv2d, bn, inp1)
    parent2 = conv(efficientnet, conv2d, bn, inp2)

    x = tf.math.add(parent1, parent2)
    x = tf.math.divide(x, 2)
    x = hub.KerasLayer("https://tfhub.dev/google/progan-128/1")(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)

    model = keras.models.Model(inputs=[inp1_full_size, inp2_full_size], outputs=x)

    return model


model = get_model()
model.summary()


def get_callbacks(train_dataset, test_dataset):

    # Create space for logs and checkpoints
    model_name = "family_gan_v1"
    log_dir = os.path.join("data", "logs", "logs_" + model_name)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join("data", "checkpoints", "checkpoints_" + model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_dir,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )

    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=5e-3,
        patience=5,
        verbose=True,
        mode="min",
        restore_best_weights=True,
    )

    return [
        checkpoint_callback,
        early_stopping_callback,
    ]


if __name__ == "__main__":

    # Load dataset
    dataset = FacesDataset("data/dataset_summary.json")
    train_dataset, test_dataset = dataset.get_train_test_dataset()

    # Get model
    model = get_model()
    model.compile(optimizer="adam", loss=MeanSquaredError())

    callbacks = get_callbacks(train_dataset, test_dataset)
    model.fit(
        train_dataset,
        epochs=100,
        steps_per_epoch=dataset.dataset_train_len // dataset.batch_size,
        validation_data=test_dataset,
        validation_steps=dataset.dataset_test_len // dataset.batch_size * 2,
        callbacks=callbacks,
    )
