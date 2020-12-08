# -Course-Recommentation-Project

## Inroduction
* This model is designed to generate course recommendation for Yonsei University, UIC students. 
* The model is based on LDA topic modeling, JS Divergence and TF-IDF score to compute documnet context similarity.
* The model takes career keyword as an input (ex. data science, business strategy, marketing..) -the model query relevant Job Description from database, and then compute similarity of these Job Description and each courses to recommend the most relevant courses
* Document similarity score are based on weighted average of JS-distance of LDA document-topic distribution, and TF-IDF scores of key-terms
* Queyring algorithm uses wordnet distance

## How to use
* Run program_main.py 
* Put your name, division, and keyword as an input
* The model displays queried job descriptions based on input keyword. Exclude irrelevant job descriptions for more precise recommendation
* The model generate 10 recommendations 

