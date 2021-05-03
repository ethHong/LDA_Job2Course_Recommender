from ast import literal_eval
from LDAModel_utils import *
import pickle

data = pd.read_csv("data/processed_courses_data.csv")
data["tokenized"] = data["tokenized"].apply(literal_eval)

dictionary, corpus, lda = train_lda(data, num_topics=40, passes=25)
pickle.dump(corpus, open("model/lda_corpus.pkl", "wb"))
pickle.dump(dictionary, open("model/lda_dictionary.pkl", "wb"))

pickle.dump(lda, open("model/lda_model.pkl", "wb"))