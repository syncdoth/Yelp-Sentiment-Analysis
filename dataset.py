import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocess import *


def create_dataloader(root,
                      mode,
                      model_name,
                      tokenizer=None,
                      batch_size=32,
                      max_length=256,
                      columns=["cool", "funny", "useful"]):
    review_ds = SentimentDataset(root,
                                 mode,
                                 model_name,
                                 tokenizer=tokenizer,
                                 max_length=max_length,
                                 columns=columns)

    # shuffle the dataset if it is not test dataset
    dataloader = torch.utils.data.DataLoader(review_ds,
                                             batch_size=batch_size,
                                             shuffle=mode != "test")

    class_weights = review_ds.get_class_weights()

    return dataloader, class_weights


class SentimentDataset(torch.utils.data.Dataset):

    def __init__(self,
                 root,
                 mode,
                 model_name,
                 framework="pt",
                 max_length=256,
                 columns=["cool", "funny", "useful"],
                 tokenizer=None):
        self.root = root
        self.mode = mode
        self.data_file = pd.read_csv(os.path.join(self.root, f"{self.mode}.csv"))
        self.framework = framework

        self.review_texts = None
        if model_name == "lstm-cnn":
            self.review_texts = self.data_file["text"].map(lower).map(tokenize).map(stem)
            if mode != "train":
                assert tokenizer is not None
                assert isinstance(tokenizer, Tokenizer)
                self.tokenizer = tokenizer
            else:
                self.tokenizer = Tokenizer(split=' ', oov_token="[OOV]")
                self.tokenizer.fit_on_texts(self.review_texts)
        else:
            if "roberta" in model_name:
                tokenizer_base = RobertaTokenizer
            elif "bert" in model_name:
                tokenizer_base = BertTokenizer
            elif "xlnet" in model_name:
                tokenizer_base = XLNetTokenizer
            else:
                raise NotImplementedError
            if mode != "train":
                assert tokenizer is not None
                assert isinstance(tokenizer, tokenizer_base)
                self.tokenizer = tokenizer
            else:
                self.tokenizer = tokenizer_base.from_pretrained(model_name)
        self.max_length = max_length

        if self.review_texts is None:
            self.review_texts = self.data_file["text"].to_list()
        if mode != "test":
            self.stars = self.data_file["stars"].to_numpy()
            self.stars -= 1  # 1~5 -> 0~4

        if len(columns) == 0:
            self.other_features = None
            return

        # normalize other features to 0~1
        self.other_features = MinMaxScaler().fit_transform(
            self.data_file[columns].to_numpy())

    def __len__(self):
        return len(self.review_texts)

    def __getitem__(self, idx):
        text = self.review_texts[idx]
        if self.mode != "test":
            label = self.stars[idx]

        encoded = self.tokenizer.encode_plus(text,
                                             add_special_tokens=True,
                                             max_length=self.max_length,
                                             return_token_type_ids=False,
                                             padding='max_length',
                                             return_attention_mask=True,
                                             return_tensors=self.framework,
                                             truncation=True)

        data = {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0]
        }
        if self.mode != "test":
            data["label"] = label

        if self.other_features is not None:
            data["features"] = torch.FloatTensor(self.other_features[idx])
        return data

    def get_class_weights(self):
        if self.mode == "test":
            return None
        return compute_class_weight('balanced',
                                    classes=np.unique(self.stars),
                                    y=self.stars)

    def get_keras_data(self):
        data = self.tokenizer.texts_to_sequences(self.review_texts)
        data = [pad_sequences(data, maxlen=self.max_length), self.other_features]

        return data, self.stars
