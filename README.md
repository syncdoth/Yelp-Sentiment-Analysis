# Yelp-Sentiment-Analysis
HKUST COMP 4332 project1 - Spring 2021 

## Usage

First, download the dataset from canvas, name the folder `data_2021_spring`, or just any name of your choice.

Call the `main.py` script. This will run the experiments with the given parameters.

To check all the parameters, type in your shell:

```
  python main.py --help
```

This will show the default parameters and explanations.

Example run code would be:

```
python main.py --data_path <data folder path>\
  --model_name bert-base-cased \
  --batch_size 16 \
  --max_len 256 \
  --epochs 5 \
  --lr 0.00002 \
  --dropout 0.2 \
  --save_path models/{}_bs{}_lr{}_drop{}.pth \
  --nouse_pooled
```

* The code assumes that `models` folder is already created. If you haven't, run `mkdir models` 
before calling above script.
