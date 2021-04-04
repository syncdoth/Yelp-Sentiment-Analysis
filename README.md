# Yelp-Sentiment-Analysis
HKUST COMP 4332 project1 - Spring 2021

## Usage

First, download the dataset from canvas, name the folder `data_2021_spring`, or just any name of your choice.

### Install
Then, install required packages.

```
pip install -r requirements.txt
```
* Recommend using virtual env (venv or conda), since both tensorflow and pytorch
will be installed. Also, you might want to finetune the versions of packages to
resolve any dependency issues.

### Train
Call the `main.py` script for train. This will run the experiments with the given parameters.

To check all the parameters, type in your shell:

```
  python main.py --help
```

This will show the default parameters and explanations.

Example (best) train code would be:

```
python main.py --data_path <data folder path> \
  --model_name roberta-base \
  --batch_size 16 \
  --max_len 256 \
  --epochs 8 \
  --lr 2e-5 \
  --dropout 0.3 \
  --save_path models/{}_bs{}_lr{}_drop{}_hidden{}.pth \
  --use_pooled \
  --other_hidden_dim 32
```
* Above HP will achieve about 0.67 validation accuracy.
* Notice that our implementation makes use of Bert or Roberta model implemented
in [hugginface](https://huggingface.co/transformers/index.html)'s `transformer`
package. They will install some pretrained model weights to
`~/.cache/huggingface` directory.

### Evaluation
For evaluation, call `evaluate.py` script.

* For validation set, it will generate a classification report.
```
python evaluate.py --data_path <data folder path> \
  --model_path <saved model checkpoint path> \
  --which_data valid
```
* For test set, it will generate a `pred.csv` file.
```
python evaluate.py --data_path <data folder path> \
  --model_path <saved model checkpoint path> \
  --which_data test \
  --save_path preds/pred.csv
```

## Keras LSTM (+CNN) model

LSTM models are implemented in keras, and run through `lstm_train.py` file.
```
python lstm_train.py
```
* Hyperparameters needs to be tuned manually within the file.
* You can decide to use pretrained embedding from
[glove](https://nlp.stanford.edu/projects/glove/). Download the desired embedding
file and change line 31 of `lstm_train.py` to the path of the glove embedding
file.
* since this model proved to be suboptimal, no inference code was written for this
model.