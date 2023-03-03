from dataclasses import dataclass
from typing import Any, Tuple
import tensorflow as tf
import numpy as np
import pandas as pd


@dataclass
class ConfigSAE:
    optimizer: Any = "adam"
    epochs: int = 200
    batch_size: int = 64
    verbose: int = 2
    input_shape: Tuple = None

@dataclass
class DataSets:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_train_md: pd.DataFrame
    x_val_md: pd.DataFrame
    x_train_pre: pd.DataFrame
    x_val_pre: pd.DataFrame


class SAEImp:
    def __init__(self):
        self._fitted = False
        self._model = None

    @staticmethod
    def _data_generator(x_train: np.ndarray, x_train_pre: np.ndarray,
                        x_train_noise: np.ndarray, batch_size: int):
        while True:
            if batch_size < x_train.shape[0]:
                idx_batch = np.random.choice(x_train.shape[0], batch_size, replace=False)
                x = [x_train_pre[idx_batch], x_train[idx_batch], x_train_noise[idx_batch]]
                y = x_train[idx_batch]
            else:
                x = [x_train_pre, x_train, x_train_noise]
                y = x_train
            yield x, y

    class _TripletLossLayer(tf.keras.layers.Layer):
        def call(self, inputs, *args, **kwargs):
            emb_size = int(inputs.shape[1] / 3)
            anchor, positive, negative = inputs[:, :emb_size], inputs[:, emb_size:2 * emb_size], inputs[:, 2 * emb_size:]
            positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
            negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
            self.add_loss(tf.reduce_mean(tf.maximum(positive_dist - negative_dist + 0.2, 0.)), inputs=inputs)
            return anchor

    @staticmethod
    def _create_cnn(config: ConfigSAE, dense_dim: int = 1024):
        emb_x = emb_input = tf.keras.Input(shape=(config.input_shape[0], ))
        emb_x = tf.keras.layers.BatchNormalization()(emb_x)
        emb_x = tf.keras.layers.Dense(dense_dim)(emb_x)
        emb_x = tf.keras.layers.Reshape((dense_dim // 16, 16))(emb_x)
        emb_x = tf.keras.layers.BatchNormalization()(emb_x)
        emb_x = tf.keras.layers.Conv1D(filters=16, kernel_size=5, activation="relu",
                                       use_bias=False, padding='SAME')(emb_x)
        emb_x = tf.keras.layers.MaxPool1D(strides=2, padding="SAME")(emb_x)
        emb_x = tf.keras.layers.BatchNormalization()(emb_x)
        emb_x = tf.keras.layers.Dropout(rate=0.25)(emb_x)
        res_conv_1 = emb_x
        emb_x = tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation="relu",
                                       use_bias=True, padding='SAME')(emb_x)
        emb_x = tf.keras.layers.Multiply()([res_conv_1, emb_x])
        emb_x = tf.keras.layers.MaxPool1D(strides=2, padding="SAME")(emb_x)
        emb_x = tf.keras.layers.BatchNormalization()(emb_x)
        emb_x = tf.keras.layers.Dropout(rate=0.25)(emb_x)
        emb_x = tf.keras.layers.Flatten()(emb_x)
        emb_x = tf.keras.layers.Activation('tanh')(emb_x)
        emb_x = tf.keras.layers.Dense(128, activation='tanh')(emb_x)

        embedding_model = tf.keras.models.Model(emb_input, emb_x)

        dec_x = dec_input = tf.keras.Input(shape=(128,))
        dec_x = tf.keras.layers.Dense(dense_dim // 4, activation='tanh')(dec_x)
        dec_x = tf.keras.layers.BatchNormalization()(dec_x)
        dec_x = tf.keras.layers.Reshape((16, 16))(dec_x)
        dec_x = tf.keras.layers.Conv1DTranspose(filters=16, kernel_size=3, strides=2, activation="relu",
                                                use_bias=True, padding='SAME')(dec_x)
        dec_x = tf.keras.layers.BatchNormalization()(dec_x)
        dec_x = tf.keras.layers.Dropout(rate=0.25)(dec_x)
        dec_x = tf.keras.layers.Conv1DTranspose(filters=16, kernel_size=5, strides=2, activation="relu",
                                                use_bias=True, padding='SAME')(dec_x)
        dec_x = tf.keras.layers.BatchNormalization()(dec_x)
        dec_x = tf.keras.layers.Dropout(rate=0.25)(dec_x)
        dec_x = tf.keras.layers.Flatten()(dec_x)
        dec_x = tf.keras.layers.Activation('tanh')(dec_x)
        dec_x = tf.keras.layers.BatchNormalization()(dec_x)
        dec_x = tf.keras.layers.Dense(config.input_shape[0], activation='sigmoid')(dec_x)

        decode_model = tf.keras.models.Model(dec_input, dec_x)
        return embedding_model, decode_model

    def fit(self, datasets: DataSets, config: ConfigSAE):
        embedding_model, decode_model = self._create_cnn(config)
        input_anchor = tf.keras.layers.Input(shape=config.input_shape)
        input_positive = tf.keras.layers.Input(shape=config.input_shape)
        input_negative = tf.keras.layers.Input(shape=config.input_shape)

        embedding_anchor = embedding_model(input_anchor)
        embedding_positive = embedding_model(input_positive)
        embedding_negative = embedding_model(input_negative)

        emb_output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)
        emb_output = self._TripletLossLayer()(emb_output)
        output = decode_model(emb_output)
        net = tf.keras.models.Model([input_anchor, input_positive, input_negative], output)
        net.compile(loss=tf.losses.mean_squared_error, optimizer=config.optimizer)

        gauss_train = np.random.normal(0.5, 0.05 ** 0.5, np.isnan(datasets.x_train_md.values).astype(int).sum())
        x_train_noise = datasets.x_train_md.values.copy()
        x_train_noise[np.isnan(datasets.x_train_md.values)] = gauss_train

        gauss_val = np.random.normal(0.5, 0.05 ** 0.5, np.isnan(datasets.x_val_md.values).astype(int).sum())
        x_val_noise = datasets.x_val_md.values.copy()
        x_val_noise[np.isnan(datasets.x_val_md.values)] = gauss_val

        bs = config.batch_size if config.batch_size < datasets.x_train.shape[0] else datasets.x_train.shape[0]
        _ = net.fit(
            x=self._data_generator(datasets.x_train.values, datasets.x_train_pre.values, x_train_noise, config.batch_size),
            validation_data=([datasets.x_val_pre.values, datasets.x_val.values, x_val_noise], datasets.x_val.values),
            batch_size=bs,
            steps_per_epoch=datasets.x_train.shape[0] // bs,
            epochs=config.epochs,
            verbose=config.verbose
        )

        self._model = tf.keras.models.Model(input_anchor, decode_model(embedding_anchor))
        self._fitted = True

    def transform(self, dataset: pd.DataFrame):
        if not self._fitted:
            raise RuntimeError("The fit method must be called before transform.")
        return self._model.predict(dataset.values)
