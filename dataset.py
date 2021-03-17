import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer


def create_dataloader(root,
                      mode,
                      model_name,
                      batch_size=32,
                      max_length=256,
                      columns=["cool", "funny", "useful"]):
    review_ds = SentimentDataset(root,
                                 mode,
                                 model_name,
                                 max_length=max_length,
                                 columns=columns)

    dataloader = torch.utils.data.DataLoader(review_ds, batch_size=batch_size)

    class_weights = review_ds.get_class_weights()

    return dataloader, class_weights


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 mode,
                 model_name,
                 max_length=256,
                 columns=["cool", "funny", "useful"]):
        self.root = root
        self.mode = mode
        self.data_file = pd.read_csv(os.path.join(self.root, f"{self.mode}.csv"))

        self.review_texts = self.data_file["text"].to_list()
        if mode != "test":
            self.stars = self.data_file["stars"].to_numpy()
            self.stars -= 1  # 1~5 -> 0~4

        if "roberta" in model_name:
            tokenizer_base = RobertaTokenizer
        elif "bert" in model_name:
            tokenizer_base = BertTokenizer
        elif "xlnet" in model_name:
            tokenizer_base = XLNetTokenizer
        self.tokenizer = tokenizer_base.from_pretrained(model_name)
        self.max_length = max_length

        self.other_features = []

        # normalize other features to 0~1
        for key in self.data_file[columns]:
            self.other_features.append(MinMaxScaler().fit_transform(
                self.data_file[columns][key].to_numpy().reshape(-1, 1)))
        self.other_features = np.concatenate(self.other_features, 1)

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
                                             return_tensors='pt',
                                             truncation=True)

        data = {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0]
        }
        if self.mode != "test":
            data["label"] = label

        other_features = self.other_features[idx]
        data["features"] = torch.FloatTensor(other_features)
        return data

    def get_class_weights(self):
        return compute_class_weight('balanced',
                                    classes=np.unique(self.stars),
                                    y=self.stars)
