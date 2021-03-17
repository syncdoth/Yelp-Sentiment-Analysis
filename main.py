import os

from absl import flags
from absl import app
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

import transformers
from transformers import (BertModel, BertTokenizer, RobertaModel,
                          RobertaTokenizer, XLNetModel, XLNetTokenizer, AdamW,
                          get_linear_schedule_with_warmup)
import torch
from torch import nn
from tqdm import tqdm

flags.DEFINE_string("data_path", "data_2021_spring", "data directory path")
flags.DEFINE_string("model_name", "bert-base-cased",
                    "which transformer to use")
flags.DEFINE_integer("batch_size", 16, "batch size: 16 or 32 preferred")
flags.DEFINE_integer("max_len", 256,
                     "max sentence length. max value is 512 for bert")
flags.DEFINE_integer("epochs", 3, "number of training epochs")
flags.DEFINE_float("lr", 2e-5, "learning rate. Preferred 2e-5, 3e-5, 5e-5")
flags.DEFINE_list("other_features", ["cool", "funny", "useful"],
                  "other feature aggregations to use")
flags.DEFINE_float("dropout", 0.3, "dropout rate")
flags.DEFINE_string("save_path", "models/{}_bs{}_lr{}_drop{}.pth",
                    "where to save the model")
flags.DEFINE_bool("use_pooled", True, "whether to use pooled output of Bert")

FLAGS = flags.FLAGS
#HP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentAnalyzer(nn.Module):
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
        self.fc1 = nn.Linear(num_other_features, hidden_size)
        self.classifier = nn.Linear(
            self.transformer.config.hidden_size + hidden_size, num_class)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input_ids, attention_mask, other_features):
        transformer_out = self.transformer(input_ids=input_ids,
                                           attention_mask=attention_mask)
        if self.use_pooled:
            output = transformer_out["pooler_output"]
        else:
            cls_token = transformer_out[
                "last_hidden_state"][:, 0]  # get the [CLS] token
            output = self.hidden(cls_token)
        dropped = self.dropout(output)  # [batch_size, 768]

        feat = self.fc1(other_features)  # [batch_size, num_other_features]

        final = torch.cat([dropped, feat], axis=1)
        return self.classifier(final)


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 mode,
                 model_name,
                 max_length=256,
                 columns=["cool", "funny", "useful"]):
        self.root = root
        self.mode = mode
        self.data_file = pd.read_csv(
            os.path.join(self.root, f"{self.mode}.csv"))

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


def create_dataloader(root,
                      mode,
                      model_name,
                      batch_size=32,
                      max_length=256,
                      columns=["cool", "funny", "useful"]):
    review_ds = Dataset(root,
                        mode,
                        model_name,
                        max_length=max_length,
                        columns=columns)

    dataloader = torch.utils.data.DataLoader(review_ds, batch_size=batch_size)

    class_weights = review_ds.get_class_weights()

    return dataloader, class_weights


def train(model, data_train, data_val, epochs, device, criterion, optimizer,
          scheduler, save_path):
    step = 0
    curr_best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(data_train, total=int(len(data_train)))

        correct_num = 0
        total_num = 0
        running_loss = 0
        for i, batch in enumerate(train_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            other_features = batch["features"].to(device)
            label = batch["label"].to(device)
            step += 1
            logits = model(input_ids, attention_mask, other_features)
            predicted = torch.max(logits, dim=1)[1]

            loss = criterion(logits, label)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            correct_num += (predicted == label).sum().item()
            total_num += label.shape[0]

            train_bar.set_postfix(acc=(correct_num / total_num),
                                  loss=(running_loss / total_num))

            del batch, input_ids, attention_mask, other_features, label, logits, loss, predicted

        print(
            f"[train] epoch: {epoch}, global step: {step}, loss: {running_loss / total_num},"
            f" accracy: {correct_num / total_num}")

        model.eval()
        y_pred = []
        y_true = []
        val_running_loss = 2
        with torch.no_grad():
            for batch in data_val:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                other_features = batch["features"].to(device)
                label = batch["label"].to(device)

                logits = model(input_ids, attention_mask, other_features)
                predicted = torch.max(logits, dim=1)[1]

                loss = criterion(logits, label)
                val_running_loss += loss.item()

                y_pred.extend(predicted.tolist())
                y_true.extend(label.tolist())

                del batch, input_ids, attention_mask, label, logits, predicted
        report = classification_report(y_true, y_pred, output_dict=True)
        print(
            f"[valid] epoch: {epoch}, global step: {step}, loss: {val_running_loss / len(data_val)},"
            f" report:\n{report}")

        if report["accuracy"] > curr_best_val_acc:
            curr_best_val_acc = report["accuracy"]
            model_dir, name = save_path.rsplit("/", 1)
            name = f"acc{curr_best_val_acc}_{name}"
            torch.save(model.state_dict(), os.path.join(model_dir, name))


def main(args):
    del args  # not used
    train_dataloader, class_weights = create_dataloader(
        FLAGS.data_path,
        "train",
        FLAGS.model_name,
        batch_size=FLAGS.batch_size,
        max_length=FLAGS.max_len,
        columns=FLAGS.other_features)
    val_dataloader, _ = create_dataloader(FLAGS.data_path,
                                          "valid",
                                          FLAGS.model_name,
                                          batch_size=FLAGS.batch_size,
                                          max_length=FLAGS.max_len,
                                          columns=FLAGS.other_features)

    model = SentimentAnalyzer(FLAGS.model_name,
                              num_class=5,
                              num_other_features=len(FLAGS.other_features),
                              dropout_rate=FLAGS.dropout,
                              use_pooled=FLAGS.use_pooled).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(
        weight=torch.FloatTensor(class_weights).to(DEVICE))
    # loss_fn = nn.CrossEntropyLoss()
    bert_optim = AdamW(model.parameters(), lr=FLAGS.lr, correct_bias=False)

    total_steps = len(train_dataloader) * FLAGS.epochs
    scheduler = get_linear_schedule_with_warmup(bert_optim,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    model_save_path = FLAGS.save_path.format(FLAGS.model_name,
                                             FLAGS.batch_size, FLAGS.lr,
                                             FLAGS.dropout)
    train(model, train_dataloader, val_dataloader, FLAGS.epochs, DEVICE,
          loss_fn, bert_optim, scheduler, model_save_path)


if __name__ == "__main__":
    app.run(main)
