import pandas as pd
import numpy as np
import os

path = os.getcwd() + "/train_dataset"

df_25 = pd.read_csv(path+"/processed_courses_data_25topic.csv")
df_40 = pd.read_csv(path+"/processed_courses_data_40topic.csv")
df_50 = pd.read_csv(path+"/processed_courses_data_50topic.csv")

df_25["topic"] = df_25[df_25.columns[3:]].idxmax(1)
df_40["topic"] = df_40[df_40.columns[3:]].idxmax(1)
df_50["topic"] = df_50[df_50.columns[3:]].idxmax(1)

df_25.to_excel("topic_25.xlsx", index = False)
df_40.to_excel("topic_40.xlsx", index = False)
df_50.to_excel("topic_50.xlsx", index = False)

def gini_impurity(list_data):
    total = list(set(list_data))
    p=0
    for i in total:
        p += ((list_data.count(i)/len(list_data)))**2

    gini = 1-p
    return gini

def get_gini_topics(df):
    gini = []
    topics = df.columns[3:-1]

    for i in topics:
        topic_df = df.loc[df["topic"]==i]
        temp_list = list(topic_df["div"].values)
        gini.append(gini_impurity(temp_list))

    return pd.DataFrame({"topic" : topics, "gini": gini})


get_gini_topics(df_25).to_excel("gini_impurity_25.xlsx", index = False)
get_gini_topics(df_40).to_excel("gini_impurity_40.xlsx", index = False)
get_gini_topics(df_50).to_excel("gini_impurity_50.xlsx", index = False)