import pandas as pd
import numpy as np
import os
from crawling_linkedin import JDcrawler_recommender
from datetime import datetime
from tqdm import tqdm

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
wn = wordnet

files = os.listdir("database")
dir = os.getcwd()+"/database"
data = [i for i in files if i.split(".")[-1]=="csv"]

df = pd.DataFrame()
for filename in data:
    filename = dir+"/"+filename
    d = pd.read_csv(filename)
    df = pd.concat([df, d])

#번역 해야함

#Cleanse Current DB

from stopword_filter import *
df.Position = df.Position.apply(lambda x: cleanse(x, False))
df.Job_Details = df.Job_Details.apply(lambda x: cleanse(x, False))


def relevance_query(a, b, wn = wn):
    #Break Coined words
    a = cleanse(a, stem_words=False).split()
    b = cleanse(b, stem_words=False).split()

    #고유명사 없애기
    a = [i for i in a if len(wn.synsets(i))>0]
    b = [i for i in b if len(wn.synsets(i))>0]

    num_distances = len(a)*len(b)

    distance = 0

    for a_i in a:
        for b_i in b:
            a_wn = wn.synsets(a_i)[0]
            b_wn = wn.synsets(b_i)[0]
            dist = a_wn.path_similarity(b_wn)
            if dist == None:
                dist =0
            distance += dist
    #0.2 이상이면 통과
    try:
        return distance/num_distances
    except:
        return 0


def query(keywords, df):
    tqdm.pandas()
    print ("Querying Result from DB...")
    df["Query_score"] = df["Position"].progress_apply(lambda x: relevance_query(x, keywords, wn))
    q = df.loc[df["Query_score"]>0.15].sort_values(by = "Query_score",ascending = False).drop("Query_score", axis = 1)
    return q

def enhance_db(current_db, keywords, rec):
    result = query(keywords, current_db)[:10].reset_index(drop = True)
    print(result[['Position', 'Job_Details']])

    react = input("Are result satisfying?: y/n")

    if react == "y":
        return result
    else:
        rec.login_linkedin()

        print("관련된 JD 와 모집공고를 추가 조회합니...(시간이 걸릴 수 있습니다!)")
        keyword = " ".join(keywords.split())
        result = rec.crawl(keyword, counts=20, how_many=3)

        # refine result

        result[result.columns[0]] = result[result.columns[0]].apply(lambda x: cleanse(x, stem_words=False))
        result[result.columns[1]] = result[result.columns[1]].apply(lambda x: cleanse(x, stem_words=True))

        result.reset_index(drop=True, inplace=True)
        result["tokenized"] = result["Job_Details"].apply(lambda x : x.split())

        update_db = pd.concat([current_db, result])
        update_db["Metatag"] = update_db["Metatag"].fillna("[]")
        update_db = update_db.drop_duplicates(["Job_Details"])
        update_db.drop_duplicates(["Job_Details"], keep = "last")
        update_db.to_csv(dir + "/Database.csv", index = False)
        print("DB를 업데이트했습니다")

        print(query(keywords, update_db)[:10].reset_index(drop = True))
        return query(keywords, update_db)[:10].reset_index(drop = True)