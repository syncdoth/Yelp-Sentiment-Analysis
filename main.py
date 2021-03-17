import os

from absl import flags
from absl import app
from sklearn.metrics import classification_report
import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from model import SentimentAnalyzer
from dataset import create_dataloader

flags.DEFINE_string("data_path", "data_2021_spring", "data directory path")
flags.DEFINE_string("model_name", "bert-base-cased", "which transformer to use")
flags.DEFINE_integer("batch_size", 16, "batch size: 16 or 32 preferred")
flags.DEFINE_integer("max_len", 256, "max sentence length. max value is 512 for bert")
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


def train(model, data_train, data_val, epochs, device, criterion, optimizer, scheduler,
          save_path):
    step = 0
    curr_best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        train_bar = tqdm(data_train, total=int(len(data_train)))

        correct_num = 0
        total_num = 0
        running_loss = 0
        for batch in train_bar:
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
        val_running_loss = 0
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
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, name))


def main(args):
    del args  # not used
    train_dataloader, class_weights = create_dataloader(FLAGS.data_path,
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

    loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(DEVICE))
    # loss_fn = nn.CrossEntropyLoss()
    bert_optim = AdamW(model.parameters(), lr=FLAGS.lr, correct_bias=False)

    total_steps = len(train_dataloader) * FLAGS.epochs
    scheduler = get_linear_schedule_with_warmup(bert_optim,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    model_save_path = FLAGS.save_path.format(FLAGS.model_name, FLAGS.batch_size, FLAGS.lr,
                                             FLAGS.dropout)
    train(model, train_dataloader, val_dataloader, FLAGS.epochs, DEVICE, loss_fn,
          bert_optim, scheduler, model_save_path)


if __name__ == "__main__":
    app.run(main)
