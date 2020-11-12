from crawling_linkedin import JDcrawler_recommender
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
import os
from fake_useragent import UserAgent
from tqdm.notebook import trange
import time
import bs4 as BeautifulSoup
from LDAModel import train_lda, jsd
from stopword_filter import stem_word, filter_stopwords, cleanse, sw, stemmer, cleanse_df, tfidf, filter_more
from ast import literal_eval

driverpath = os.getcwd()+"/chromedriver"
options = webdriver.ChromeOptions()
driver =  webdriver.Chrome(driverpath,  chrome_options=options)

recommender = JDcrawler_recommender("홍석현", driverpath, driver, options)
data = recommender.load_processed()
data["tokenized"] = data["tokenized"].apply(literal_eval)


keyword = recommender.keyword

recommender.login_linkedin()

print ("관련된 JD 와 모집공고를 조회중입니다...(시간이 걸릴 수 있습니다!)")
result = recommender.crawl(keyword, counts = 5, how_many=3)

#refind result


result[result.columns[0]] = result[result.columns[0]].apply(lambda x: cleanse(x, stem_words = False))
result[result.columns[1]] = result[result.columns[1]].apply(lambda x: cleanse(x, stem_words = False))

result.reset_index(drop = True, inplace = True)

print (result)

first = input("Choose index of first JD you are interested: ")
second = input("Choose index of second JD you are interested: ")
third = input("Choose index of third JD you are interested: ")

target = result.iloc[[int(first), int(second), int(third)]]
target_doc = target["Job_Details"].values.sum().split()

def process_filter(target):
    target = " ".join([i.lower() for i in target])
    target = filter_stopwords(target, sw)
    target = cleanse(target)
    target = target.split()

    return target

target_doc = process_filter(target_doc)

print ("내용을 기반으로 추천을 생성하는 중입니다...1~2분정도 소요될 수 있습니다")

from LDAModel import train_lda, jsd
dictionary,corpus,lda = train_lda(data, num_topics = int(recommender.topicnum), passes = 25)



doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
query_bow = dictionary.doc2bow(target_doc)
new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=query_bow)])
index_to_score = jsd(new_doc_distribution, doc_topic_dist)
data["distance_with_JD"] = index_to_score

result = data[["Course_Name", "tokenized", "div", "distance_with_JD"]].sort_values(by = "distance_with_JD")[:10]

print (result)