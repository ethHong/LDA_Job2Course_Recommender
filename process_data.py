import pandas as pd
import numpy as np
import gensim
from gensim.models import LdaModel
from gensim import models, corpora
from gensim.models.coherencemodel import CoherenceModel
from LDAModel_utils import *
from tqdm import tqdm


print("Loading data...")
df = pd.read_csv("data/courses_data.csv")
df = df.loc[df["Year"].isin([2018, 2019, 2020])]
print("Data Loaded!...")
print("Transforming target doc and collections...")

df = df.loc[df["Division"].apply(lambda x: x.startswith("언더우드국제대학"))]
df = cleanse_df(df)

additional = filter_more(df, 0.1)["words"].values

for i in additional:
    sw.append(i)


def more(token):
    return [i for i in token if i not in sw]


tqdm.pandas()
df["tokenized"] = df["tokenized"].progress_apply(lambda x: more(x))

# Make bigrams


df = df.drop_duplicates("Course_Name", keep="last")
df.to_csv("data/processed_courses_data.csv", index=False)
