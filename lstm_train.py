from dataset import SentimentDataset
from model import get_lstm_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle
import numpy as np
from preprocess import get_pretrained_embedding

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"  # select gpu
MAX_LEN = 512
DATA_PATH = "data_2021_spring"
DROPOUT = 0.3
OTHER_HIDDEN = 32
LSTM_HIDDEN = 128
LR = 5e-5

train_dataset = SentimentDataset(DATA_PATH,
                                 "train",
                                 "lstm-cnn",
                                 max_length=MAX_LEN,
                                 columns=["cool", "funny", "useful"],
                                 framework="tf")
val_dataset = SentimentDataset(DATA_PATH,
                               "valid",
                               "lstm-cnn",
                               max_length=MAX_LEN,
                               columns=["cool", "funny", "useful"],
                               framework="tf",
                               tokenizer=train_dataset.tokenizer)

train_data, train_label = train_dataset.get_keras_data()
val_data, val_label = val_dataset.get_keras_data()

embedding_matrix = get_pretrained_embedding("glove.42B.300d.txt", train_dataset.tokenizer,
                                            300)

lstm_model = get_lstm_model(len(train_dataset.tokenizer.word_index) + 1,
                            num_class=5,
                            num_other_features=3,
                            max_seq_len=MAX_LEN,
                            embed_dim=300,
                            embed_weight=embedding_matrix,
                            hidden_size=LSTM_HIDDEN,
                            other_size=OTHER_HIDDEN,
                            dropout_rate=DROPOUT)


# Callbacks: lr schedule and checkpoint
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


lr_schedule = keras.callbacks.LearningRateScheduler(scheduler)

checkpoint_filepath = "models/models"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True)

lstm_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)])

class_weight = dict(enumerate(train_dataset.get_class_weights()))
history = lstm_model.fit(train_data,
                         train_label,
                         validation_data=(val_data, val_label),
                         batch_size=16,
                         epochs=50,
                         class_weight=class_weight,
                         callbacks=[lr_schedule, model_checkpoint_callback])
