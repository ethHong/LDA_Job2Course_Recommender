import pandas as pd
import json
import re
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm, tqdm_pandas
from ast import literal_eval

num_topics = int(input("Put your topic number"))
passes = 25

data = pd.read_csv("processed_courses_data_{}topic.csv".format(num_topics))
data["tokenized"] = data["tokenized"].apply(literal_eval)

print ("Loading Target Documents")

print ("Load JD of Google..")
with open('linkedin_JD_Google.json') as json_file:
    json_data = json.load(json_file)
print ("Load JD of BCG..")
with open('linkedin_JD_Boston Consulting Group.json') as json_file:
    json_data2 = json.load(json_file)
print ("Load JD of JP Morgan..")
with open('linkedin_JD_jpmorgan.json') as json_file:
    json_data3 = json.load(json_file)

print ("Transforming target doc and collections...")

collection = list(json_data.values())
target_doc =[keyword for bag in collection for keyword in bag]
position  =[i for i in list(json_data.keys())]

collection2 = list(json_data2.values())
target_doc2 =[keyword for bag in collection2 for keyword in bag]
position2  =[i for i in list(json_data2.keys())]

collection3 = list(json_data3.values())
target_doc3 =[keyword for bag in collection3 for keyword in bag]
position3  =[i for i in list(json_data3.keys())]

print ("Filtering Stopwords from target docs...")
from stopword_filter import stem_word, filter_stopwords, cleanse, sw, stemmer, cleanse_df, tfidf, filter_more

def process_filter(target):
    target = " ".join([i.lower() for i in target])
    target = filter_stopwords(target, sw)
    target = cleanse(target)
    target = target.split()

    return target

target_doc = process_filter(target_doc)
target_doc2 = process_filter(target_doc2)
target_doc3 = process_filter(target_doc3)

print ("Train Test Split")
from sklearn.model_selection import train_test_split
validation = pd.read_csv("LDA_Result_3company_{}Topics.csv".format(num_topics))
train, test = train_test_split(data, test_size=0.2)

print ("Training Baseline LDA")

from LDAModel import train_lda, jsd
dictionary,corpus,lda = train_lda(train, num_topics = num_topics, passes = 25)
