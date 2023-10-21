import pandas as pd
import spacy
from typing import List
from tqdm import tqdm

nlp = spacy.load("en_core_web_md")
df = pd.read_csv("data/raw/filtered.tsv", sep="\t")

# for each entry find difference between translation and reference lemma wise


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


df_toxic = get_toxic_vocab(df)

# save as csv
df_toxic.to_csv("data/interim/toxic_words.csv", index=False)
