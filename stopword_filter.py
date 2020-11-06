import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


import time

nltk.download('stopwords')
sw = stopwords.words("english")
sw+=["hw", "assignment", "overview", "review", "assign", "introduce", "introduction", "introduct", "course", "syllabus", "test", "exam", "ch", "chapter", "lecture", "class"] #manual하게 일부 단어
stemmer = PorterStemmer()


def stem_word(string):
    string = [stemmer.stem(i) for i in string.split()]
    string = [i for i in string if len(i) > 1]
    return " ".join(string)


for i in sw:
    if stem_word(i) not in sw:
        sw.append(stem_word(i))


def filter_stopwords(doc, sw):
    filtered = [i for i in doc.split() if i not in sw]
    return " ".join(filtered)


def cleanse(string):
    try:
        cleansed = " ".join(string.rsplit("\\n"))
        cleansed = re.sub('[^A-Za-z]+', ' ', cleansed).lower()
        cleansed = filter_stopwords(cleansed, sw)

        cleansed = stem_word(cleansed)

        return cleansed
    except:
        return ""


def cleanse_df(df):
    temp = df

    temp["detail"] = temp["syllabus"] + " " + temp["course_info"]
    temp["detail"] = temp["detail"].apply(lambda x: cleanse(x))
    temp["detail"] = temp["detail"].apply(lambda x: cleanse(x))

    temp['tokenized'] = temp['detail'].apply(lambda x: x.split()).apply(lambda x: "None" if x == [] else x)
    temp["div"] = temp["Code"].apply(lambda x: re.sub('[^A-Za-z]+', '', x))

    output = temp[["Course_Name", "tokenized", "div"]]
    output = output.loc[output["tokenized"] != "None"]
    return output


def tfidf(data, how="sum"):
    corpus = [" ".join(i) for i in data["tokenized"]]
    tfidf_voctorize = TfidfVectorizer().fit(corpus)

    if how == "sum":
        avg_score = tfidf_voctorize.transform(corpus).toarray().sum(0)
    elif how == "avg":
        avg_score = tfidf_voctorize.transform(corpus).toarray().mean(0)

    vocab = tfidf_voctorize.get_feature_names()

    scores = pd.DataFrame({"words": vocab, "scores": avg_score}).sort_values(by="scores", ascending=False)
    return scores


def filter_more(data, threshold=0.005, par="sum"):
    tfidf_df = tfidf(data, par)

    scores = list(set(tfidf_df["scores"]))
    scores.sort()

    filter_thred_score = scores[int(threshold * len(scores))]
    print("total vocabs: {}".format(tfidf_df.shape[0]))

    return tfidf_df.loc[tfidf_df["scores"] < filter_thred_score]