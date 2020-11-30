from crawling_crossval import JDcrawler_recommender_crossval
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import os
from fake_useragent import UserAgent
from tqdm import tqdm_gui
import time
import bs4 as BeautifulSoup
from LDAModel import train_lda, jsd
from stopword_filter import stem_word, filter_stopwords, cleanse, sw, stemmer, cleanse_df, tfidf, filter_more
from ast import literal_eval
dir = os.getcwd()+"/database"
def process_filter(target):
    target = " ".join([i.lower() for i in target])
    target = filter_stopwords(target, sw)
    target = cleanse(target)
    target = target.split()
    return target
keyword = input("Validation Keyword: ")

driverpath = os.getcwd()+"/chromedriver"
options = webdriver.ChromeOptions()
driver =  webdriver.Chrome(driverpath,  chrome_options=options)

recommender = JDcrawler_recommender_crossval("홍석현", driverpath, driver, options, 40, keyword)
data = recommender.load_processed()
data["tokenized"] = data["tokenized"].apply(literal_eval)

from process_DB import query, enhance_db

include_meta = input("Include Metadata?: y/n")

DB = pd.read_csv(dir+"/DB_updated_metatags_40thred.csv").dropna()
#DB = pd.read_csv(dir+"/DB_updated_metatags.csv").dropna()
#DB = pd.read_csv(dir+"/Database.csv").dropna()


DB["tokenized"] = DB["tokenized"].apply(literal_eval)
DB["Metatag"] = DB["Metatag"].apply(literal_eval)

print ("DB에서 관련 Job Description 을 조회중입니다...")
result = query(keyword, DB).reset_index(drop = True)
print (result)
first = input("첫번째 테스트 인풋을 넣어주세요")
second = input("두번째 테스트 인풋을 넣어주세요")
third = input("세번째 테스트 인풋을 넣어주세요")

target = result.iloc[[int(first), int(second), int(third)]]
target_doc = target["Job_Details"].values.sum().split()
target_doc = process_filter(target_doc)

distances =[]
for i in tqdm_gui(range(10)):
    print("{}th iteration".format(i))

    if include_meta=="y":
        for j in target["Metatag"].values:
            for i in j:
                target_doc.append(i)



    print ("내용을 기반으로 추천을 생성하는 중입니다...1~2분정도 소요될 수 있습니다")

    from LDAModel import train_lda, jsd
    dictionary,corpus,lda = train_lda(data, num_topics = int(recommender.topicnum), passes = 25)

    doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
    query_bow = dictionary.doc2bow(target_doc)

    new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=query_bow)])
    index_to_score = jsd(new_doc_distribution, doc_topic_dist)
    data["distance_with_JD"] = index_to_score

    result = data[["Course_Name", "tokenized", "div", "distance_with_JD"]].sort_values(by = "distance_with_JD")[:10]

    print ("평균 JSD 거리는 다음과 같습니다")
    print (result["distance_with_JD"].mean())
    distances.append(result["distance_with_JD"].mean())

if include_meta=="y":
    print ("Result with Meta")
else:
    print("Result without Meta")

print ("최종 오차:")
print (distances)
print ("평균 오차:")
print (np.mean(distances))