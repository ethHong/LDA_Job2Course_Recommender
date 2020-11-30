import pandas as pd
import numpy as np
import os
from ast import literal_eval
from tqdm import tqdm, tqdm_gui, tqdm_pandas
from stopword_filter import tfidf


def load_data(topic_num=40):
    processed = pd.read_csv(
        os.getcwd() + "/train_dataset" + "/processed_courses_data_{}topic.csv".format(topic_num))


    return processed

def get_meta_tag(data, threshold = 40):
    tags= []
    scores = tfidf(data)
    keywords = scores.loc[scores["scores"]>threshold].words.values

    for i in tqdm_gui(range(len(data["tokenized"].values))):
        jd = data["tokenized"].values[i]
        temp =[]
        for term in jd:
            if term in keywords:
                temp.append(term)
        tags.append(list(set(temp)))

    return tags

