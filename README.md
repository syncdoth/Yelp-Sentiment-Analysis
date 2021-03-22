# Yelp-Sentiment-Analysis
HKUST COMP 4332 project1 - Spring 2021 

## Usage

First, download the dataset from canvas, name the folder `data_2021_spring`, or just any name of your choice.

Call the `main.py` script for train. This will run the experiments with the given parameters.

To check all the parameters, type in your shell:

```
  python main.py --help
```

This will show the default parameters and explanations.

Example train code would be:

```
python main.py --data_path <data folder path> \
  --model_name roberta-base \
  --batch_size 16 \
  --max_len 256 \
  --epochs 5 \
  --lr 2e-5 \
  --dropout 0.2 \
  --save_path models/{}_bs{}_lr{}_drop{}.pth \
  --nouse_pooled
```

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
