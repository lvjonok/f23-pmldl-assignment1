import pandas as pd
import spacy
from typing import List
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    print("Downloading NLTK Stopwords")
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

from nltk.stem.snowball import SnowballStemmer

st = SnowballStemmer("english")
nlp = spacy.load("en_core_web_md")


def get_lemmas(text: str) -> List[str]:
    """
    get_lemmas returns a list of lemmas of a given text
    """

    # lower case all strings
    text = text.lower()

    return text.split()

    # # tokenize
    # doc = nlp(text)

    # # get lemmas
    # lemmas = [token.lemma_ for token in doc]

    # return lemmas


def lemma_diff(l1: List[str], l2: List[str]) -> List[str]:
    """
    lemma_diff returns which lemmas were removed in translation compared to reference
    """

    return list(set(l1) - set(l2))


def get_toxic_vocab(df):
    # store toxic words
    toxic_words = []

    # for each entry find difference between translation and reference lemma wise and store it in df_toxic

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # skip the row if it is not toxic (0.9 is the threshold)
        if row["ref_tox"] < 0.9 or row["trn_tox"] > 0.1:
            continue

        diff = lemma_diff(get_lemmas(row["reference"]), get_lemmas(row["translation"]))

        toxic_words.extend(diff)

    # remove duplicates and save as pandas dataframe
    return pd.DataFrame(list(set(toxic_words)), columns=["toxic_words"])


def clean_data(df, col, clean_col):
    """
    clean data removes punctuation, stopwords and gets the stem of the words in the given column

    Example:
        df = clean_data(df, "toxic_words", "clean")
    """
    # change to lower and remove spaces on either side
    df[clean_col] = df[col].apply(lambda x: str(x).lower().strip())

    # remove extra spaces in between
    df[clean_col] = df[clean_col].apply(lambda x: re.sub(" +", " ", x))

    # remove punctuation
    df[clean_col] = df[clean_col].apply(lambda x: re.sub("[^a-zA-Z]", " ", x))

    # remove stopwords and get the stem
    df[clean_col] = df[clean_col].apply(
        lambda x: " ".join(
            st.stem(text) for text in x.split() if text not in stop_words
        )
    )

    return df


if __name__ == "__main__":
    # load data
    df = pd.read_csv("data/raw/filtered.tsv", sep="\t")
    df_toxic = get_toxic_vocab(df)

    # save as csv
    df_toxic.to_csv("data/interim/toxic_words.csv", index=False)
