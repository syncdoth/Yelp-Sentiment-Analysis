from dataset import SentimentDataset
from model import get_lstm_model
import tensorflow as tf
from tensorflow import keras

import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_dataset = SentimentDataset("new_data",
                                 "train",
                                 "bert-base-cased",
                                 max_length=256,
                                 columns=["cool", "funny", "useful"],
                                 framework="tf")
val_dataset = SentimentDataset("new_data",
                               "valid",
                               "bert-base-cased",
                               max_length=256,
                               columns=["cool", "funny", "useful"],
                               framework="tf")

lstm_model = get_lstm_model(train_dataset.tokenizer.vocab_size,
                            num_class=5,
                            num_other_features=3,
                            max_seq_len=256,
                            embed_dim=1024,
                            hidden_size=128,
                            other_size=10,
                            dropout_rate=0.3)

lstm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)])

train_data = [
    tf.stack([data["input_ids"] for data in train_dataset], axis=0),
    tf.stack([data["features"] for data in train_dataset], axis=0)
]
train_label = tf.concat([data["label"] for data in train_dataset], axis=0)

val_data = [
    tf.stack([data["input_ids"] for data in val_dataset], axis=0),
    tf.stack([data["features"] for data in val_dataset], axis=0)
]
val_label = tf.concat([data["label"] for data in val_dataset], axis=0)

history = lstm_model.fit(train_data,
                         train_label,
                         validation_data=(val_data, val_label),
                         batch_size=16,
                         epochs=10)

lstm_model.save("models/lstm+cnn_model_other10_hidden256.h5")
with open("models/lstm+cnn_history_other10_hidden256.pickle", "wb") as f:
    pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)
