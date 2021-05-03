from flask import render_template, request, jsonify, session, Flask
from flask_session import Session
from flask_cors import CORS, cross_origin
import pickle
import os
from ast import literal_eval
import numpy as np
import pandas as pd
from LDAModel_utils import *
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from commands import create_tables
from extensions import db
from models import User_rating

# Test Alpha Value, alpha=1 on TFIDF
alpha = 0.5

app = Flask(__name__)
app.debug = True
app.config.from_pyfile("settings.py")

db.init_app(app)

Session(app)
CORS(app)

# Create DB
engine = db.create_engine("mysql+mysqldb://root:@localhost/", {})  # connect to server
engine.execute("CREATE DATABASE IF NOT EXISTS userdatabase")  # create db

# Create Tables on DB
app.cli.add_command(create_tables)


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/home")
def home():
    return render_template("index.html")


@app.route("/selection", methods=["GET", "POST"])
def select():
    print("load JD data")
    DB = pd.read_csv("data/Translated_JD_cleansed.csv").dropna()
    print("loaded!")
    DB["Position_raw"] = DB.Position
    DB["Position"] = DB["Position_cleanse"]
    DB["Job_Details_raw"] = DB.Job_Details
    DB["Job_Details"] = DB["Job_Details_cleanse"]
    DB["tokenized_raw"] = DB["Job_Details"].apply(lambda x: x.split())
    DB["tokenized"] = DB["Job_Details_tokenized"].apply(literal_eval)

    str_features = [str(x) for x in request.form.values()]
    print(str_features)

    username = str_features[0]
    keyword = str_features[1]
    division = str_features[2]
    test_type = str_features[3]

    session["user"] = username
    session["keyword"] = keyword
    session["division"] = division
    session["test_type"] = test_type

    print("Querying Relevant JD from DB...")

    keywords = " ".join(gensim.utils.simple_preprocess(keyword, deacc=True))
    temp_tf_matrix = tfidf_matrix(DB, tokenized="tokenized_raw", name="Position")
    target = query(DB, keywords, temp_tf_matrix)

    # Check if there's no result
    if target.shape[0] == 0:
        return render_template("error.html")

    else:
        tf_idf_list = tfidf(target)

        def get_keywords(tokens, tf_idf_list):
            wordbag = list(set(tokens))
            scores = []
            for i in wordbag:
                scores.append(
                    tf_idf_list.loc[tf_idf_list["words"] == i]["scores"].values[0]
                )
            key = pd.DataFrame({"word": wordbag, "score": scores}).sort_values(
                by="score", ascending=False
            )
            key = key.loc[key["score"] >= 0.5]
            return key["word"].values

        keywords = []
        for tokens in target["tokenized"].values:
            keywords.append(get_keywords(tokens, tf_idf_list))

        target["Keywords"] = keywords
        jobs = target[
            [
                "Position",
                "Position_raw",
                "Job_Details_raw",
                "Keywords",
                "Job_Details",
                "tokenized",
            ]
        ]

        def up_all(arr):
            new = []
            for i in arr:
                new.append(i.upper())
            return np.array(new)

        jobs["Position"] = jobs["Position"].apply(lambda x: x.upper())
        jobs["Keywords"] = jobs["Keywords"].apply(lambda x: up_all(x))

        jobs["Position_raw"] = jobs["Position_raw"].apply(lambda x: x.upper())

        session["jobs"] = jobs

        JDs = []
        for i in range(0, jobs[["Position_raw", "Job_Details_raw"]].shape[0]):
            JDs.append(dict(jobs[["Position_raw", "Job_Details_raw"]].iloc[i]))
        session["JDs"] = JDs

    return render_template("select.html", result_1=JDs)


@app.route("/prediction", methods=["GET", "POST"])
def predict():
    course_data = pd.read_csv("data/course_info.csv")
    course_data["syllabus_key"] = course_data["syllabus_key"].apply(literal_eval)

    data = pd.read_csv("data/processed_courses_data.csv")
    data["tokenized"] = data["tokenized"].apply(literal_eval)

    username = session.get("username")
    keyword = session.get("keyword")
    division = session.get("division")
    test_type = session.get("test_type")
    jobs = session.get("jobs")
    JDs = session.get("JDs")

    keywords = " ".join(gensim.utils.simple_preprocess(keyword, deacc=True))
    index_features = request.form.getlist("mycheckbox")
    index_features = [int(i) - 1 for i in index_features]
    print("Checkup:", index_features)
    target = jobs.iloc[index_features]

    print("Checkup:", target)

    target_doc = target["Job_Details"].values.sum().split()
    target_doc = " ".join([i.lower() for i in target_doc])
    target_doc = remove_stopwords(target_doc.split())
    target_doc = gensim.utils.simple_preprocess(" ".join(target_doc), deacc=True)

    total_divs = np.unique(data["div"].values)
    if division == "None":
        data_filter = total_divs
    else:
        data_filter = divcats[division]
    indexer = data["div"].isin(data_filter)

    # index 를 가져오기
    data = data[indexer]
    courses = data["Course_Name"].values

    # 미리 로드한 모델 - 모든 수업으로 LDA모델링
    dictionary = pickle.load(open("model/lda_dictionary.pkl", "rb"))
    corpus = pickle.load(open("model/lda_corpus.pkl", "rb"))
    lda = pickle.load(open("model/lda_model.pkl", "rb"))

    doc_topic_dist = np.array([[tup[1] for tup in lst] for lst in lda[corpus]])
    doc_topic_dist = doc_topic_dist[indexer]
    query_bow = dictionary.doc2bow(target_doc)
    new_doc_distribution = np.array(
        [tup[1] for tup in lda.get_document_topics(bow=query_bow)]
    )

    index_to_score_JSD = jsd(new_doc_distribution, doc_topic_dist)
    index_to_score_KL = KL(new_doc_distribution, doc_topic_dist)

    data["distance_with_JD_JSD"] = index_to_score_JSD
    data["distance_with_KL"] = index_to_score_KL

    # min_max_scaler = MinMaxScaler()
    # min_max_scaler.fit(data["distance_with_JD_JSD"].values.reshape(-1, 1))
    # scaled_JSD = min_max_scaler.transform(data["distance_with_JD_JSD"].values.reshape(-1, 1))

    # min_max_scaler.fit(data["distance_with_KL"].values.reshape(-1, 1))
    # scaled_KL = min_max_scaler.transform(data["distance_with_KL"].values.reshape(-1, 1))

    weight = alpha
    k = 5
    # TF-IDF 스코어
    jd_tfidf = tfidf(target)
    jd_keywords = jd_tfidf["words"].values[:k]  # Expanded Keyword
    tf_matrix = tfidf_matrix(data)
    vector = np.array([0] * tf_matrix.shape[1])

    for i in jd_keywords:
        if i in tf_matrix.index:
            vector = vector + tfidf_matrix(data).loc[i].values

    data["TF-IDF_keyscore"] = vector
    data["distance_with_JD_JSD"] = data["distance_with_JD_JSD"].apply(
        lambda x: 1 / (x + 0.001)
    )

    # minmax
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(data["TF-IDF_keyscore"].values.reshape(-1, 1))
    data["TF-IDF_keyscore"] = min_max_scaler.transform(
        data["TF-IDF_keyscore"].values.reshape(-1, 1)
    )

    min_max_scaler.fit(data["distance_with_JD_JSD"].values.reshape(-1, 1))
    data["distance_with_JD_JSD"] = min_max_scaler.transform(
        data["distance_with_JD_JSD"].values.reshape(-1, 1)
    )

    # data["distance_with_KL"] = data["distance_with_KL"].apply(lambda x: 1 / (x + 0.001))
    data["Aggregate_score"] = (
        data["distance_with_JD_JSD"] * (1 - weight) + data["TF-IDF_keyscore"] * weight
    )
    result = data[
        ["Course_Name", "tokenized", "div", "Aggregate_score", "course_info"]
    ].sort_values(by="Aggregate_score", ascending=False)
    result = result[:10]
    result = result.reset_index(drop=True)

    output = result[["Course_Name", "div", "Aggregate_score", "course_info"]]
    output.columns = [
        "Recommended Course",
        "Division",
        "Recommendation Score1",
        "Course Description",
    ]

    print("Checkup1: ", output)
    # if test_type=="A":
    #    output.to_csv(os.getcwd() + "/app/test_output/recommender_1_alpha_{}.csv".format(alpha), index = False)
    min_max_scaler = MinMaxScaler((0, 10))
    min_max_scaler.fit(output["Recommendation Score1"].values.reshape(-1, 1))
    scaled_score1 = min_max_scaler.transform(
        output["Recommendation Score1"].values.reshape(-1, 1)
    )
    output["Recommendation Score1"] = scaled_score1.round(2)

    output = output.drop(["Recommendation Score1"], axis=1)

    """info = []
    for course in output["Recommended Course"].values:

        temp_info = course_data.loc[course_data["Course_Name"] == course][
            "course_info"
        ].values[-1]
        info.append(temp_info)

    syl = []
    for course in output["Recommended Course"].values:

        temp_syl = course_data.loc[course_data["Course_Name"] == course][
            "syllabus_key"
        ].values[-1]
        temp_syl = [i for i in temp_syl if i not in sw]
        syl.append(temp_syl)

    output["Course_info"] = info
    # output["syllabus"] = syl"""

    results = []
    for i in range(0, output.shape[0]):
        results.append(dict(output.iloc[i]))

    session["rating1"] = results

    # Comparison Model - TF-IDF score

    vector2 = np.array([0] * tf_matrix.shape[1])

    for i in gensim.utils.simple_preprocess(keywords, deacc=True):
        if i in tf_matrix.index:
            vector2 = vector2 + tfidf_matrix(data).loc[i].values
    data["comparison_score"] = vector2

    extracted = data.sort_values(by="comparison_score", ascending=False)
    extracted["Course_Name"] = extracted["Course_Name"].apply(
        lambda x: x.split("\xa0")[0]
    )

    extracted = extracted.loc[extracted["comparison_score"] > 0]
    min_max_scaler = MinMaxScaler((0, 5))
    min_max_scaler.fit(extracted["comparison_score"].values.reshape(-1, 1))
    extracted["comparison_score"] = min_max_scaler.transform(
        extracted["comparison_score"].values.reshape(-1, 1)
    )
    # Comparison only with rating score
    # extracted["final_score"] = extracted["comparison_score"] + extracted["Rating_Score"]
    # extracted["final_score"] = extracted["Rating_Score"]
    extracted["final_score"] = extracted["comparison_score"]

    result2 = extracted.sort_values(by="final_score", ascending=False)[
        :10
    ].reset_index()[["Course_Name", "div", "final_score"]]
    result2.columns = ["Recommended Course", "Division", "Recommendation Score2"]

    min_max_scaler = MinMaxScaler((0, 1))
    min_max_scaler.fit(result2["Recommendation Score2"].values.reshape(-1, 1))
    scaled_score2 = min_max_scaler.transform(
        result2["Recommendation Score2"].values.reshape(-1, 1)
    )
    result2["Recommendation Score2"] = scaled_score2.round(2)
    print(
        "Checkup 2: ", result2.sort_values(ascending=False, by="Recommendation Score2")
    )
    # if test_type=="B":
    #    result2.to_csv(os.getcwd() + "/app/test_output/recommender_2.csv", index = False)

    result2 = result2.drop(["Recommendation Score2"], axis=1)

    info = []
    for course in result2["Recommended Course"].values:
        temp_info = course_data.loc[course_data["Course_Name"] == course][
            "course_info"
        ].values[-1]
        info.append(temp_info)

    syl = []
    for course in result2["Recommended Course"].values:
        temp_syl = course_data.loc[course_data["Course_Name"] == course][
            "syllabus_key"
        ].values[-1]
        temp_syl = [i for i in temp_syl if i not in sw]
        syl.append(temp_syl)

    result2["Course_info"] = info
    # result2["syllabus"] = syl
    results2 = []
    for i in range(0, result2.shape[0]):
        results2.append(dict(result2.iloc[i]))

    session["rating2"] = results2

    if test_type == "A":
        result = results
    elif test_type == "B":
        result = results2

    session["result_df"] = result
    questions = [
        "How is the recommendation relevant to your keyword?",
        "How are you satisfied with the recommendation?",
    ]
    session["questions"] = questions
    return render_template("result.html", result_1=result, questions=questions)


@app.route("/submit", methods=["GET", "POST"])
def submit_rating():
    username = session.get("user")
    keyword = session.get("keyword")
    division = session.get("division")
    test_type = session.get("test_type")
    result = session.get("result_df")
    questions = session.get("questions")

    for question in questions:
        print("Checkup1 questions: ", question)
        if test_type == "A":
            recommend = session.get("rating1")
            rating = [
                int(i)
                for i in request.form.getlist("submitted_rating_{}".format(question))
            ]
            engine_type = "LDA JD Recommender"

        elif test_type == "B":
            recommend = session.get("rating2")
            rating = [
                int(i)
                for i in request.form.getlist("submitted_rating_{}".format(question))
            ]
            engine_type = "Rating Keyword Recommender"

        for i in range(0, len(recommend)):
            recommend[i]["User"] = username
            recommend[i]["Input_keyword"] = keyword
            recommend[i]["User_division"] = division
            recommend[i]["Engine"] = engine_type
            recommend[i]["Rating"] = rating[i]
            recommend[i]["Question"] = question

        print("Checkup2 recommend: ", len(recommend), rating)

        # Add data on DB
        def add_data_on_DB(recommend):
            for row in recommend:
                created_data = User_rating(
                    user=row["User"],
                    keyword=row["Input_keyword"],
                    major=row["User_division"],
                    rating=row["Rating"],
                    question=row["Question"],
                    recommended_course=row["Recommended Course"],
                    division=row["Division"],
                    interview=None,
                    engine=row["Engine"],
                )

                db.session.add(created_data)
                db.session.commit()

        add_data_on_DB(recommend)

    return render_template("interview_question.html", result_1=result)


@app.route("/interview", methods=["GET", "POST"])
def submit_interview():
    username = session.get("user")
    keyword = session.get("keyword")
    division = session.get("division")
    test_type = session.get("test_type")

    if test_type == "A":
        recommend = session.get("rating1")
        answer = request.form.getlist("txtComments")[0]
        engine_type = "LDA JD Recommender"

    elif test_type == "B":
        recommend = session.get("rating2")
        answer = request.form.getlist("txtComments")[0]
        engine_type = "Rating Keyword Recommender"

    created_data = User_rating(
        user=username,
        keyword=keyword,
        major=division,
        question="Interview",
        rating=None,
        recommended_course=None,
        division=None,
        interview=answer,
        engine=engine_type,
    )

    db.session.add(created_data)
    db.session.commit()

    return render_template("submitted.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
