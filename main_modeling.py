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

data = pd.read_csv("train_dataset/processed_courses_data_{}topic.csv".format(num_topics))
data["tokenized"] = data["tokenized"].apply(literal_eval)

print ("Loading Target Documents")
print ("Load JD of Operation..")
jd_operation = pd.read_csv("JD_operation.csv")
print ("Load JD of Strategy..")
jd_strategy = pd.read_csv("JD_Strategy.csv")
print ("Load JD of Data Analysis..")
jd_data = pd.read_csv("JD_data.csv")

print ("Transforming target doc and collections...")

collection = list(jd_operation["Job_Details"].values)
target_doc = []
for i in collection:
    for j in i.split():
        target_doc.append(j)
position  =[i for i in list(jd_operation.Position.values)]

collection2 = list(jd_strategy["Job_Details"].values)
target_doc2 = []
for i in collection2:
    for j in i.split():
        target_doc2.append(j)
position2  =[i for i in list(jd_strategy.Position.values)]

collection3 = list(jd_data["Job_Details"].values)
target_doc3 = []
for i in collection3:
    for j in i.split():
        target_doc3.append(j)
position3  =[i for i in list(jd_data.Position.values)]

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

print ("Training Baseline LDA")

from LDAModel import train_lda, jsd
dictionary,corpus,lda = train_lda(data, num_topics = num_topics, passes = 25)

print ("showing topics: ")
lda.show_topics(num_topics=10, num_words=10)

print ("Moeling - Operation")
doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
query_bow = dictionary.doc2bow(target_doc)
new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=query_bow)])
index_to_score = jsd(new_doc_distribution, doc_topic_dist)
data["distance_Operation"] = index_to_score

print ("Moeling - Strategy")
query_bow2 = dictionary.doc2bow(target_doc2)
new_doc_distribution2 = np.array([tup[1] for tup in lda.get_document_topics(bow=query_bow2)])
index_to_score2 = jsd(new_doc_distribution2, doc_topic_dist)
data["distance_Strategy"] = index_to_score2

print ("Moeling - Data Analysis")
query_bow3 = dictionary.doc2bow(target_doc3)
new_doc_distribution3 = np.array([tup[1] for tup in lda.get_document_topics(bow=query_bow3)])
index_to_score3 = jsd(new_doc_distribution3, doc_topic_dist)
data["distance_Data"] = index_to_score3

data["variance"] = data[["distance_Operation", "distance_Strategy", "distance_Data"]].var(axis = 1)

print(data)

print ("Export Data")
data = data.drop_duplicates(subset='Course_Name', keep="last")
data = data[["Course_Name", "tokenized", "div", "distance_Operation", "distance_Strategy", "distance_Data", "variance"]]

print (data)
data.to_csv("LDA_Result_3company_{}Topics_newJD.csv".format(num_topics), index = False)
data.to_excel("LDA_Result_3company_{}Topics_newJD.xlsx".format(num_topics), index = False)

