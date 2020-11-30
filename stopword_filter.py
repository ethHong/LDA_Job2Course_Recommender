import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


import time

nltk.download('stopwords')
sw = stopwords.words("english")
sw+=["li", "ul", "span", "hw", "assignment", "overview", "review", "assign", "introduce", "introduction", "introduct", "course", "syllabus", "test", "exam", "ch", "chapter", "lecture", "class"] #manual하게 일부 단어
stemmer = PorterStemmer()


def stem_word(string):
    string = [stemmer.stem(i) for i in string.split()]
    string = [i for i in string if len(i) > 1]
    return " ".join(string)


for i in sw:
    if stem_word(i) not in sw:
        sw.append(stem_word(i))


def filter_stopwords(doc, sw):
    for i in sw:
        if stem_word(i) not in sw:
            sw.append(stem_word(i))
    filtered = [i for i in doc.split() if i not in sw]
    return " ".join(filtered)


def cleanse(string, stem_words = True):
    try:
        cleansed = " ".join(string.rsplit("\\n"))
        cleansed = re.sub('[^A-Za-z]+', ' ', cleansed).lower()
        cleansed = filter_stopwords(cleansed, sw)

        if stem_words == True:
            cleansed = stem_word(cleansed)

        return cleansed
    except:
        return ""


def cleanse_df(df, stem_words = True):
    temp = df

    temp["detail"] = temp["syllabus"] + " " + temp["course_info"]
    temp["detail"] = temp["detail"].apply(lambda x: cleanse(x, stem_words))
    temp["detail"] = temp["detail"].apply(lambda x: cleanse(x, stem_words))

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


from sklearn import preprocessing
def tfidf_matrix(data):
    corpus = [" ".join(i) for i in data["tokenized"]]
    tfidf_voctorize = TfidfVectorizer().fit(corpus)

    avg_score = tfidf_voctorize.transform(corpus).toarray().T
    vocab = tfidf_voctorize.get_feature_names()
    courses =data["Course_Name"].values
    avg_score = preprocessing.minmax_scale(avg_score.T).T
    scores = pd.DataFrame(avg_score, index = vocab, columns=courses)
    return scores

def filter_more(data, threshold=0.005, par="sum"):
    tfidf_df = tfidf(data, par)

    scores = list(set(tfidf_df["scores"]))
    scores.sort()

    filter_thred_score = scores[int(threshold * len(scores))]
    print("total vocabs: {}".format(tfidf_df.shape[0]))

    return tfidf_df.loc[tfidf_df["scores"] < filter_thred_score]


divterms = { #Humanity 는 특히 구별을 못하고 있기 때문에 culture / humanities, Law 등 전공 키워드를 추가함
'ASP':"Asian Culture Humanity Studies",
'BIO': "Biology",
'BIZ':"Business",
'BTE': "Biology Tech Engineering",
'CDM':"Culture Design Management",
'CEE': "Construction Engineering",
'CHE': "Chemistry",
'CLC': "Comparitive Literature",
'CNT': "Fashion Design",
'CTM': "Creative Technology Management",
'DAA':"Chemistry, Biochemical Engineering",
'DSN' : "Design",
'ECO' : "Economics",
'EEE':"Electronic Engineering",
'ELL' :"English Literature",
'ESE' : "Energy Engineering",
'IID' : "Interactive Information Design",
'ISM' :"International Studies",
'JCL' : "Justice Civil Leadership, Law, Legal",
'LSB' : " Life System Bio Engineering, Biochemical" ,
'LST' : "Life Science Technology, Biology",
'MAT':"Mathmatics",
'MEU':"Mechanical Engineering",
'MST':"Material Engineering",
'NSE': "Nano Science Engineering",
'PHY': "Philosophy",
'POL': "Politics, Goverment Policy",
'PSY': "Psychology",
'PUB': "Public Policy, Goverment Policy",
'QRM': "Quantitative Risk Management",
'SDC': "Sustainable Development",
'SED': "Nano Science, Biology, Energy",
'SOC' :"Social Studies",
'STA': "Statistics",
'STP': "Science Technolgoy Poliocy",
'STS': "Science Technology",
'TAP': "Technology Art, ",
'UBC':"Biology, Bio Engineering",
'UIC': "Liberal Art",
'YCA': "Religion, Christianity"
}
divcats = {
    "CLC": ["CLC", "ELL"],
    "ECO" : ["ECO", "STA"],
    "ISM" : ["ISM", "JCL", "SDC"],
    "POL" : ["POL"],
    "LSB" : ["BTE", "LSB"],
    "ASP" : ["ASP", "CLC", "POL", "BIZ", "SOC"],
    "IID" : ["IID", "CDM", "CTM"],
    "CTM" : ["CTM", "CDM", "STP", "IID"],
    "CDM" : ["CDM", "DSN", "TAP"],
    "JCL" : ["JCL", "ISM", "POL"],
    "QRM" :["QRM", "STA", "ECO"],
    "STP" :["STP", "CTM", "PUB", "SDC", "JCL"],
    "SDC" :["SDC", "ISM", "STP"],
    "NSE" : ["NSE", "SED", "MAT", "SED", "STA", "EEE", "IID", "MST", "DAA","ESE"],
    "ESE" : ["ESE", "MAT", "NSE", "SED", "STA", "DAA", "CEE", "IID", "MEU"],
    "SED" : ["SED", "UBC", "LSB", "STA", "NSE"]
}