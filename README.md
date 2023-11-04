# Lev Kozlov - B20-RO

Email: l dot kozlov at innopolis dot university

## Requirements:

- I have tested the code on Ubuntu distribution with python 3.10
- Freezed requirements are in [requirements.txt](requirements.txt)
- Install with `pip install -r requirements.txt`

## Structure:

See [PROBLEM.md](PROBLEM.md)

Although all results are shown in reports, I have created [notebooks](notebooks/) with aim to show my work process alongside with the code.

The report about baseline and how I have approached the problem is [here](reports/solution_building.md)

The report about my final model and overall evaluation is [here](reports/final_solution.md)

## Usage:

### Data processing:

1. [data/make_dataset.py](src/data/make_dataset.py) - downloads the dataset and extracts it into `data/raw` directory.
2. [data/split_dataset.py](src/data/split_dataset.py) - takes the raw dataset and splits into train and val parts. Moreover, it fixes 500 entries for faster evaluation.
3. [data/make_toxic_set.py](src/data/make_toxic_set.py) - script to find set difference between translation and original sentence. It is used to create the vocabulary of toxic words.

### Train model:

Training can be done in [this script](src/models/train_model.py)

It is assumed that you have ran the data processing scripts before.

### Predict:

Prediction can be done in [this script](src/models/predict_model.py)

The script will load the model from `models` directory and transform the argument sentence.

Example usage:

```bash
python3 src/models/predict_model.py --input "Get out of my fucking house"
```
