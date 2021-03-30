from transformers import BertModel, RobertaModel, XLNetModel
import torch
from torch import nn
from tensorflow import keras
import tensorflow as tf


class TransformerSentimentAnalyzer(nn.Module):

    def __init__(self,
                 model_name,
                 num_class=5,
                 num_other_features=3,
                 hidden_size=10,
                 dropout_rate=0.3,
                 use_pooled=True):
        super().__init__()
        self.use_pooled = use_pooled
        if "roberta" in model_name:
            transformer_base = RobertaModel
        elif "bert" in model_name:
            transformer_base = BertModel
        elif "xlnet" in model_name:
            transformer_base = XLNetModel
            self.use_pooled = False  # no pooler for xlnet
        self.transformer = transformer_base.from_pretrained(model_name)
        if not self.use_pooled:
            self.hidden = nn.Linear(self.transformer.config.hidden_size,
                                    self.transformer.config.hidden_size)
            nn.init.xavier_uniform(self.hidden.weight)
        if num_other_features > 0:
            self.fc1 = nn.Linear(num_other_features, hidden_size)
            nn.init.xavier_uniform(self.fc1.weight)
            self.other_relu = nn.ReLU()
            self.classifier = nn.Linear(self.transformer.config.hidden_size + hidden_size,
                                        num_class)
        else:
            self.classifier = nn.Linear(self.transformer.config.hidden_size, num_class)
        nn.init.xavier_uniform(self.classifier.weight)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input_ids, attention_mask, other_features):
        transformer_out = self.transformer(input_ids=input_ids,
                                           attention_mask=attention_mask)
        if self.use_pooled:
            output = transformer_out["pooler_output"]
        else:
            cls_token = transformer_out["last_hidden_state"][:, 0]  # get the [CLS] token
            output = self.hidden(cls_token)
        dropped = self.dropout(output)  # [batch_size, 768]
        if hasattr(self, "fc1"):
            feat = self.fc1(other_features)  # [batch_size, num_other_features]
            feat = self.other_relu(feat)
            final = torch.cat([dropped, feat], axis=1)
        else:
            final = dropped
        return self.classifier(final)


def get_lstm_model(vocab_size,
                   num_class=5,
                   num_other_features=3,
                   max_seq_len=256,
                   embed_dim=256,
                   hidden_size=256,
                   other_size=10,
                   dropout_rate=0.3):
    """Tensorflow lstm-cnn model"""
    x = keras.Input((max_seq_len,))
    emb = keras.layers.Embedding(input_dim=vocab_size,
                                 output_dim=embed_dim)(x)  # [batch, T, E]
    output = keras.layers.Bidirectional(
        keras.layers.LSTM(hidden_size, return_sequences=True,
                          dropout=dropout_rate))(emb)  # [batch, T, H * 2]
    output = keras.layers.Conv1D(hidden_size,
                                 2,
                                 input_shape=output.shape[1:],
                                 activation="relu")(output)
    output = tf.keras.layers.MaxPooling1D(pool_size=output.shape[1])(output)
    output = keras.layers.Flatten()(output)
    output = keras.layers.Dropout(rate=dropout_rate)(output)

    if num_other_features > 0:
        other = keras.Input((num_other_features,))
        other_feat = keras.layers.Dense(other_size, activation="relu")(other)
        output = tf.concat([output, other_feat], 1)

    output = keras.layers.Dense(num_class, activation=None)(output)

    if num_other_features > 0:
        return keras.Model(inputs=[x, other], outputs=output)
    return keras.Model(inputs=x, outputs=output)
