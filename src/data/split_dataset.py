import pandas as pd


def filter_dataset(df: pd.DataFrame, t1: float, t2: float):
    """
    filter_dataset removes from original dataset all the rows where
    reference toxicity is less than t1 and translation toxicity is greater than t2
    """

    # filter out rows where reference toxicity is less than t1
    df = df[df["ref_tox"] > t1]

    # filter out rows where translation toxicity is greater than t2
    df = df[df["trn_tox"] < t2]

    return df


def split_dataset(df: pd.DataFrame, ratio: float):
    """
    split dataset divides the dataset into two parts based on the given ratio
    """

    # get the number of rows to be taken in the first part
    n = int(len(df) * ratio)

    # shuffle the dataset
    df = df.sample(frac=1)

    # take the first n rows
    df1 = df.iloc[:n]

    # take the rest
    df2 = df.iloc[n:]

    return df1, df2


if __name__ == "__main__":
    # load data
    df = pd.read_csv("data/raw/filtered.tsv", sep="\t")

    # filter dataset
    df = filter_dataset(df, 0.9, 0.1)

    # split dataset
    df1, df2 = split_dataset(df, 0.8)

    # save as csv
    df1.to_csv("data/interim/train.csv", index=False)
    df2.to_csv("data/interim/val.csv", index=False)

    # save subset of validation dataset with only 500 entries
    df2 = df2.sample(n=500)
    df2.to_csv("data/interim/val_subset.csv", index=False)
