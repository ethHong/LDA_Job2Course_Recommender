import time
import gensim
from gensim.models import LdaModel
from gensim import models, corpora
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from tqdm import tqdm


def train_lda(data, num_topics=10, chunksize=500, alpha="auto", eta="auto", passes=2):
    num_topics = num_topics
    chunksize = chunksize
    dictionary = corpora.Dictionary(data["tokenized"])
    corpus = [dictionary.doc2bow(doc) for doc in data["tokenized"]]
    t1 = time.time()
    lda = LdaModel(
        corpus=corpus,
        num_topics=40,
        id2word=dictionary,
        alpha=alpha,
        eta=eta,
        chunksize=chunksize,
        minimum_probability=0.0,
        passes=passes,
    )
    t2 = time.time()
    print(
        "Time to train LDA model on ",
        data.shape[0],
        "articles: ",
        (t2 - t1) / 60,
        "min",
    )
    return dictionary, corpus, lda


def tuning_passes(data):
    print("Initializing with baseling: numtopic 10, chunksize 500")
    dictionary, corpus, lda = train_lda(data)
    coherences = []
    perplexities = []
    passes = []

    for i in range(25):
        ntopics, nwords = 200, 100
        if i == 0:
            p = 1
        else:
            p = i * 2
        tic = time.time()
        tuninglda = LdaModel(
            corpus, id2word=dictionary, num_topics=ntopics, iterations=400, passes=p
        )
        print("epoch", p, time.time() - tic)
        cm = CoherenceModel(model=tuninglda, corpus=corpus, coherence="u_mass")
        coherence = cm.get_coherence()
        print("Coherence", coherence)
        coherences.append(coherence)
        print("Perplexity: ", tuninglda.log_perplexity(corpus), "\n\n")
        passes.append(p)
        perplexities.append(tuninglda.log_perplexity(corpus))

    plt.plot(passes, coherences)
    plt.xlabel("Passes")
    plt.ylabel("Coherence")
    plt.show()

    plt.plot(passes, perplexities)
    plt.xlabel("Passes")
    plt.ylabel("Perplexity")
    plt.show()


def tuning_topics(data, p):
    dictionary, corpus, lda = train_lda(data)
    coherencesT = []
    perplexitiesT = []
    ntopic = []

    for i in range(20):
        if i == 0:
            ntopics = 2
        else:
            ntopics = 10 * i
        nwords = 100
        tic = time.time()
        lda4 = models.ldamodel.LdaModel(
            corpus, id2word=dictionary, num_topics=ntopics, iterations=400, passes=p
        )
        print("ntopics", ntopics, time.time() - tic)

        cm = CoherenceModel(model=lda4, corpus=corpus, coherence="u_mass")
        coherence = cm.get_coherence()
        print("Coherence", coherence)
        coherencesT.append(coherence)
        print("Perplexity: ", lda4.log_perplexity(corpus), "\n\n")
        perplexitiesT.append(lda4.log_perplexity(corpus))
        ntopic.append(ntopics)

    plt.plot(ntopic, coherencesT)
    plt.show()

    plt.plot(ntopic, perplexitiesT)
    plt.show()


def jsd(query, matrix):
    scores = []
    for i in range(0, matrix.shape[0]):
        p = query
        q = matrix.T[:, i]
        m = 0.5 * (p + q)

        jensen = np.sqrt(0.5 * (entropy(p, m)) + 0.5 * entropy(q, m))
        scores.append(jensen)

    return scores


def KL(query, matrix):
    scores = []
    for i in range(0, matrix.shape[0]):
        p = query
        q = matrix.T[:, i]

        KL = entropy(p, q)
        scores.append(KL)

    return scores


nltk.download("stopwords")
sw = stopwords.words("english")
sw += [
    "li",
    "ul",
    "tbd",
    "span",
    "hw",
    "assignment",
    "overview",
    "review",
    "assign",
    "introduce",
    "introduction",
    "introduct",
    "course",
    "syllabus",
    "test",
    "exam",
    "ch",
    "chapter",
    "lecture",
    "class",
]  # manual하게 일부 단어


def remove_stopwords(texts):
    return [w for w in texts if w not in sw]


def remove_distracts(texts):
    text = [re.sub("\S*@\S*\s?", "", sent) for sent in texts.split()]
    text = [re.sub("\s+", " ", sent) for sent in text]


def cleanse_df(df):
    print("cleansing dataframe...")
    temp = df[["Course_Name", "course_info", "syllabus", "Code"]]
    temp["detail"] = temp["syllabus"] + " " + temp["course_info"]
    temp = temp[["Course_Name", "syllabus", "course_info", "detail", "Code"]].dropna(
        axis=0
    )
    temp["detail"] = temp["detail"].apply(lambda x: x.replace("\\n", ""))

    temp["tokenized"] = temp["detail"].apply(
        lambda x: gensim.utils.simple_preprocess(x, deacc=True)
    )
    temp["tokenized"] = temp["tokenized"].apply(lambda x: remove_stopwords(x))

    temp["div"] = temp["Code"].apply(lambda x: re.sub("[^A-Za-z]+", "", x))

    output = temp[["Course_Name", "course_info", "syllabus", "tokenized", "div"]]
    output = output.loc[output["tokenized"] != "None"]
    return output


def tfidf(data, how="sum", tokenized="tokenized"):
    corpus = [" ".join(i) for i in data[tokenized]]
    tfidf_voctorize = TfidfVectorizer().fit(corpus)

    if how == "sum":
        avg_score = tfidf_voctorize.transform(corpus).toarray().sum(0)
    elif how == "avg":
        avg_score = tfidf_voctorize.transform(corpus).toarray().mean(0)

    vocab = tfidf_voctorize.get_feature_names()

    scores = pd.DataFrame({"words": vocab, "scores": avg_score}).sort_values(
        by="scores", ascending=False
    )
    return scores


from sklearn import preprocessing


def tfidf_matrix(data, tokenized="tokenized", name="Course_Name"):
    corpus = [" ".join(i) for i in data[tokenized]]
    tfidf_voctorize = TfidfVectorizer().fit(corpus)

    avg_score = tfidf_voctorize.transform(corpus).toarray().T
    vocab = tfidf_voctorize.get_feature_names()
    courses = data[name].values
    avg_score = preprocessing.minmax_scale(avg_score.T).T
    scores = pd.DataFrame(avg_score, index=vocab, columns=courses)
    return scores


def filter_more(data, threshold=0.005, par="sum"):
    tfidf_df = tfidf(data, par)

    scores = list(set(tfidf_df["scores"]))
    scores.sort()

    filter_thred_score = scores[int(threshold * len(scores))]
    print("total vocabs: {}".format(tfidf_df.shape[0]))

    return tfidf_df.loc[tfidf_df["scores"] < filter_thred_score]


def tfidf_score(tf_matrix, keyword):
    vector = np.array([0] * tf_matrix.shape[1])
    for i in keyword.split():
        if i in tf_matrix.index:
            vector = vector + tf_matrix.loc[i].values
    return vector


def process_filter(target):
    target = " ".join([i.lower() for i in target])
    target = filter_stopwords(target, sw)
    target = gensim.utils.simple_preprocess(target)
    target = target.split()

    return target


divterms = {  # Humanity 는 특히 구별을 못하고 있기 때문에 culture / humanities, Law 등 전공 키워드를 추가함
    "ASP": "Asian Culture Humanity Studies",
    "BIO": "Biology",
    "BIZ": "Business",
    "BTE": "Biology Tech Engineering",
    "CDM": "Culture Design Management",
    "CEE": "Construction Engineering",
    "CHE": "Chemistry",
    "CLC": "Comparitive Literature",
    "CNT": "Fashion Design",
    "CTM": "Creative Technology Management",
    "DAA": "Chemistry, Biochemical Engineering",
    "DSN": "Design",
    "ECO": "Economics",
    "EEE": "Electronic Engineering",
    "ELL": "English Literature",
    "ESE": "Energy Engineering",
    "IID": "Interactive Information Design",
    "ISM": "International Studies",
    "JCL": "Justice Civil Leadership, Law, Legal",
    "LSB": " Life System Bio Engineering, Biochemical",
    "LST": "Life Science Technology, Biology",
    "MAT": "Mathmatics",
    "MEU": "Mechanical Engineering",
    "MST": "Material Engineering",
    "NSE": "Nano Science Engineering",
    "PHY": "Philosophy",
    "POL": "Politics, Goverment Policy",
    "PSY": "Psychology",
    "PUB": "Public Policy, Goverment Policy",
    "QRM": "Quantitative Risk Management",
    "SDC": "Sustainable Development",
    "SED": "Nano Science, Biology, Energy",
    "SOC": "Social Studies",
    "STA": "Statistics",
    "STP": "Science Technolgoy Poliocy",
    "STS": "Science Technology",
    "TAP": "Technology Art, ",
    "UBC": "Biology, Bio Engineering",
    "UIC": "Liberal Art",
    "YCA": "Religion, Christianity",
}
divcats = {
    "CLC": ["CLC", "ELL"],
    "ECO": ["ECO", "STA"],
    "ISM": ["ISM", "JCL", "SDC"],
    "POL": ["POL"],
    "LSB": ["BTE", "LSB"],
    "ASP": ["ASP", "CLC", "POL", "BIZ", "SOC"],
    "IID": ["IID", "CDM", "CTM"],
    "CTM": ["CTM", "CDM", "STP", "IID"],
    "CDM": ["CDM", "DSN", "TAP"],
    "JCL": ["JCL", "ISM", "POL"],
    "QRM": ["QRM", "STA", "ECO"],
    "STP": ["STP", "CTM", "PUB", "SDC", "JCL"],
    "SDC": ["SDC", "ISM", "STP"],
    "NSE": ["NSE", "SED", "MAT", "SED", "STA", "EEE", "IID", "MST", "DAA", "ESE"],
    "ESE": ["ESE", "MAT", "NSE", "SED", "STA", "DAA", "CEE", "IID", "MEU"],
    "SED": ["SED", "UBC", "LSB", "STA", "NSE"],
}


def query(df, keywords, tf_matrix):
    tqdm.pandas()
    print("Querying Result from DB...")
    keywords = " ".join(gensim.utils.simple_preprocess(keywords, deacc=True))
    df["Query_score"] = tfidf_score(tf_matrix, keywords)
    q = (
        df.loc[df["Query_score"] > 0]
        .sort_values(by="Query_score", ascending=False)
        .drop("Query_score", axis=1)
    )
    print("If the result is not successful, please proceed to enhabce DB")
    result = q[:10].reset_index(drop=True)
    print(result)
    return result
