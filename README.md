# -Course-Recommentation-Project

***This one is the initial version of this model. Since the final version is on user test and paper project, I temporarily turned the final version private***

## Introduction
* This model is designed to generate course recommendation for Yonsei University, UIC students. 
* The model is based on LDA topic modeling, JS Divergence and TF-IDF score to compute documnet context similarity.
* The model takes career keyword as an input (ex. data science, business strategy, marketing..) -the model query relevant Job Description from database, and then compute similarity of these Job Description and each courses to recommend the most relevant courses
* Document similarity score are based on weighted average of JS-distance of LDA document-topic distribution, and TF-IDF scores of key-terms

## How to use

### app.py
* Requirements: **MysqlDB - Python connection / gensim / pickle / python **
* run app.py to run recommendation interface. It will run on local server
* Enter name, and keyword of your career interest. Recommend putting industry / Positions / Skill - wise keywords
* If you are Yonsei UIC student, put your major - if not, choose "None (Not a UIC Student)"
* (Type selection is for A/B test, but for not it's only set as 'Default')
* Push 'Generate recommendation', and it will query relevant job descriptions - Choose 3 of your interest
* Based on Job Descriptions you choose, it generate top 10 courses

### **train_model.py** and **LDAModel_utils**
* LDAModel_utils contain all the function to model LDA, and process text data
* train_model.py train LDA based on optimal parameter - 40 topics and 25 passes, and export as pkl. file
* If you want to train LDA model with custom settings, run train_model.py
