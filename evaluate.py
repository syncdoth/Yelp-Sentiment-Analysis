import os

from absl import flags
from absl import app
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

from model import TransformerSentimentAnalyzer
from dataset import create_dataloader

flags.DEFINE_string("data_path", "data_2021_spring", "data directory path")
flags.DEFINE_integer("max_len", 256, "max sentence length. max value is 512 for bert")
flags.DEFINE_list("other_features", ["cool", "funny", "useful"],
                  "other feature aggregations to use")
flags.DEFINE_string("model_path", None, "where to save the model", required=True)
flags.DEFINE_bool("use_pooled", True, "whether to use pooled output of Bert")
flags.DEFINE_string("save_path", "preds/pred.csv", "name of the file to save predictions")
flags.DEFINE_string("which_data", "test", "which data to evaluate on. valid or test")

FLAGS = flags.FLAGS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, test_data, device, mode="test", save_name="pred.csv"):
    test_bar = tqdm(test_data, total=int(len(test_data)))

    model.eval()
    preds = []
    if mode == "valid":
        y_true = []
    with torch.no_grad():
        for batch in test_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            other_features = batch["features"].to(device) if "features" in batch else None
            logits = model(input_ids, attention_mask, other_features)
            predicted = torch.max(logits, dim=1)[1]
            preds.extend(predicted.tolist())
            if mode == "valid":
                y_true.extend(batch["label"].tolist())

    if mode == "valid":
        print(classification_report(y_true, preds))
    else:
        review_ids = test_data.dataset.data_file["review_id"]
        save_preds(review_ids, np.array(preds), save_name)


def save_preds(review_ids, preds, save_name="pred.csv"):
    answer_df = pd.DataFrame(data={
        'review_id': review_ids,
        'stars': preds + 1,
    })
    answer_df.to_csv(save_name, index=False)


def main(args):
    del args  # unused

    ckpt_name = os.path.basename(FLAGS.model_path)
    ckpt_name = ckpt_name.rsplit(".", 1)[0]
    _, model_name, batch_size, _, dropout = ckpt_name.split("_")
    batch_size = int(batch_size[2:])
    dropout = float(dropout[4:])

    test_dataloader, _ = create_dataloader(FLAGS.data_path,
                                           FLAGS.which_data,
                                           model_name,
                                           batch_size=batch_size,
                                           max_length=FLAGS.max_len,
                                           columns=FLAGS.other_features)

    model = TransformerSentimentAnalyzer(model_name,
                              num_class=5,
                              num_other_features=len(FLAGS.other_features),
                              dropout_rate=dropout,
                              use_pooled=FLAGS.use_pooled).to(DEVICE)
    model.load_state_dict(torch.load(FLAGS.model_path))
    model.eval()

    evaluate(model,
             test_dataloader,
             DEVICE,
             mode=FLAGS.which_data,
             save_name=FLAGS.save_path)


if __name__ == "__main__":
    app.run(main)
