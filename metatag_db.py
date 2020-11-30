import pandas as pd
import numpy as np
import os
from stopword_filter import *
from tagging_data import get_meta_tag
from tqdm import tqdm, tqdm_gui
from bs4 import BeautifulSoup


print ("Loading Data...")
dir = os.getcwd()+"/database"

DB = pd.read_csv(dir+"/Database.csv").dropna()
metadb =  pd.read_csv(dir+"/Kaggle_Data_metasource.csv").dropna()[['Query', 'Description']]
metadb[metadb.columns[0]] = metadb[metadb.columns[0]].apply(lambda x: cleanse(x, stem_words=False))

#중복포지션이 많아 포지션별로 100개만 랜덤추출
positions = np.unique(metadb.Query.values)
meta_sampled = pd.DataFrame()
for p in positions:
    select = metadb.loc[metadb["Query"]==p]
    select = select.sample(100).reset_index(drop=True)
    meta_sampled = pd.concat([meta_sampled, select])

meta_sampled = meta_sampled.reset_index(drop=True)

#Tag를 달아 미리 키워드 추출
tqdm.pandas()
meta_sampled["tokenized"] = meta_sampled["Description"].progress_apply(lambda x : cleanse(x, stem_words=True))
meta_sampled["tokenized"] = meta_sampled["tokenized"].progress_apply(lambda x : x.split())

print ("Use LDA, TFIDF on DB to tag metadata...")

output = get_meta_tag(meta_sampled, threshold = 50)
meta_sampled["metatags"] = output
###DB와 metaDB 의 LDA를 통해 태그 달기
from LDAModel import train_lda, jsd

DB["tokenized"] = DB["Job_Details"].progress_apply(lambda x : cleanse(x, stem_words=True).split())
dictionary, corpus, lda = train_lda(DB, num_topics=40, passes=25)

tags = []
for i in range(DB.shape[0]):
    tags.append([])

count=1
for i in tqdm_gui(range(len(meta_sampled["metatags"].values))):
    jd = meta_sampled["metatags"].values[i]
    total = len(meta_sampled["metatags"].values)
    print("{} Out of {}...".format(count, total))

    metatags = []
    target_doc=jd
    doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
    query_bow = dictionary.doc2bow(target_doc)
    new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=query_bow)])
    index_to_score = jsd(new_doc_distribution, doc_topic_dist)
    indexes = []
    for ind in range(len(index_to_score)):
        if index_to_score[ind]<0.4:
            for keyword in jd:
                tags[ind].append(keyword)

    count+=1

tags_refine = []
for i in tags:
    tags_refine.append(list(set(i)))

DB["Metatag"] = tags_refine

DB.to_csv(dir+"/DB_updated_metatags.csv", index = False)