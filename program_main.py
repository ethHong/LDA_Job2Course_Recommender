import os
import sys
from ast import literal_eval
import numpy as np
import pandas as pd
from selenium import webdriver
from crawling_linkedin import JDcrawler_recommender
from stopword_filter import filter_stopwords, cleanse, sw, divterms, divcats, tfidf_matrix, tfidf
from sklearn.preprocessing import MinMaxScaler

platform = sys.platform

if platform == "win32":
    driverpath = os.getcwd() + "/chromedriver_win"
else:
    driverpath = os.getcwd() + "/chromedriver"

options = webdriver.ChromeOptions()
driver = webdriver.Chrome(driverpath, chrome_options=options)

username = input("User Name")
recommender = JDcrawler_recommender(username, driverpath, options, driver)

data = recommender.load_processed()

data["tokenized"] = data["tokenized"].apply(literal_eval)

from process_DB import dir, enhance_db, relevance_query, wn

include_meta = input("Include Metadata?: y/n")

if include_meta == "y":
    # DB = pd.read_csv(dir+"/DB_updated_metatags_40thred.csv").dropna()
    DB = pd.read_csv(dir + "/DB_updated_metatags.csv").dropna()
else:
    DB = pd.read_csv(dir + "/Database.csv").dropna()

DB["tokenized"] = DB["tokenized"].apply(literal_eval)
DB["Metatag"] = DB["Metatag"].apply(literal_eval)

print("DB에서 관련 Job Description 을 조회중입니다...")
keyword = recommender.keyword
keywords = cleanse(keyword, stem_words=False)

print ("INPUT: {}".format(keywords))

result = enhance_db(DB, keywords, recommender)
exclude = input("Exclude Irrelevant (comma seperated):, Enter if all valid")

# target = result.iloc[[int(first), int(second), int(third)]]
# 10개모두사용
if exclude == "":
    target = result
else:
    exclude = [int(i) for i in exclude.split(",")]
    target = result.loc[~result.index.isin(exclude)]

key_positions = target["Position"].values
target_doc = target["Job_Details"].values.sum().split()

if include_meta == "y":
    for j in target["Metatag"].values:
        for i in j:
            target_doc.append(i)


def process_filter(target):
    target = " ".join([i.lower() for i in target])
    target = filter_stopwords(target, sw)
    target = cleanse(target)
    target = target.split()

    return target


target_doc = process_filter(target_doc)
data_filter = divcats[recommender.division]


data =data[data["div"].isin(data_filter)]

print("내용을 기반으로 추천을 생성하는 중입니다...1~2분정도 소요될 수 있습니다")

from LDAModel import train_lda, jsd, KL

dictionary, corpus, lda = train_lda(data, num_topics=int(recommender.topicnum), passes=25)
doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
query_bow = dictionary.doc2bow(target_doc)
new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=query_bow)])

courses = data["Course_Name"].values
print ("Creating Course Network Matrix...")

def course_dist_matrix(courses, doc_topic_dist):
    matrix = []
    for i in doc_topic_dist:
        matrix.append(jsd(i, doc_topic_dist))
    return pd.DataFrame(matrix, index = courses, columns= courses)

print ("Computing JSD and KL Score...")

course_net_matrix = course_dist_matrix(courses, doc_topic_dist)
index_to_score_JSD = jsd(new_doc_distribution, doc_topic_dist)
index_to_score_KL = KL(new_doc_distribution, doc_topic_dist)

data["distance_with_JD_JSD"] = index_to_score_JSD
data["distance_with_KL"] = index_to_score_KL

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(data["distance_with_JD_JSD"].values.reshape(-1, 1))
scaled_JSD = min_max_scaler.transform(data["distance_with_JD_JSD"].values.reshape(-1, 1))

min_max_scaler.fit(data["distance_with_KL"].values.reshape(-1, 1))
scaled_KL = min_max_scaler.transform(data["distance_with_KL"].values.reshape(-1, 1))

weight = 10
print ("Computing TF-IDF score with weight {}...".format(weight))
#TF-IDF 스코어
jd_tfidf = tfidf(target)
jd_keywords = jd_tfidf["words"].values[:5]
tf_matrix = tfidf_matrix(data)
vector =np.array([0]*tf_matrix.shape[1])

for i in jd_keywords:
    if i in tf_matrix.index:
        vector = vector + tfidf_matrix(data).loc[i].values

data["TF-IDF_keyscore"]= vector
data["distance_with_JD_JSD"] = data["distance_with_JD_JSD"].apply(lambda x: 1/(x+0.001))
data["distance_with_KL"] = data["distance_with_KL"].apply(lambda x: 1/(x+0.001))

print ("Computed Aggregate Score!")

data["Aggregate_score"] = data["distance_with_JD_JSD"] + data["distance_with_KL"] + data["TF-IDF_keyscore"]*weight


result = data[["Course_Name", "tokenized", "div", "Aggregate_score"]].sort_values(by="Aggregate_score", ascending = False)
result = result[:10]

if result.shape[0] > 10:
    result = result[:10]
result =result.reset_index(drop = True)

# result["Course_Name_cleansed"] = result["Course_Name"].apply(lambda x: cleanse(x))

def get_relevant_courses(course_net_matrix, result):
    courses = result["Course_Name"].values
    matrix = course_net_matrix[courses]
    relevant = {}
    for i in courses:
        sorted = matrix.sort_values(by=i)[[i]]
        relevant[i] = list(sorted.index[1:6])
    return relevant

raw = pd.read_csv(os.getcwd() + "/train_dataset/Courses_raw_names.csv")

details = []
for i in result["Course_Name"].values:
    try:
        details.append(raw.loc[raw["Course_Name"] == i]["course_info"].values[0])
    except:
        details.append("Unidentified")

result["Details"] = details

print ("Final Recommendation: ")
print ("Main Recommendation: ")
print (result)

print ("Relevant Recommendation: ")
relevant = get_relevant_courses(course_net_matrix, result)
print (relevant)
relevant = pd.DataFrame({"Main": list(relevant.keys()), "Relevant": list(relevant.values())})
relevant.to_excel(os.getcwd() + "/User_Test/{}Relevant_suggestion.xlsx".format(username), index=False)
result.to_excel(os.getcwd() + "/User_Test/{}_suggestion.xlsx".format(username), index=False)
print("Relevant Positions: ")
print(key_positions)
