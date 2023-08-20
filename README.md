# LDA and Job Description based Course Recommendation System


## Introduction
* The model is based on LDA topic modeling, JS Divergence and TF-IDF score to compute documnet context similarity.
* The model takes career keyword as an input (ex. data science, business strategy, marketing..) -the model query relevant Job Description from database, and then compute similarity of these Job Description and each courses to recommend the most relevant courses
* Document similarity score are based on weighted average of JS-distance of LDA document-topic distribution, and TF-IDF scores of key-terms

## How to use

### app.py
* Requirements: **MysqlDB - Python connection / gensim / pickle / nltk **
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

![model](https://user-images.githubusercontent.com/43837843/116873208-3fa4e900-ac52-11eb-834c-3ed6bc63ead0.gif)

## About the project
### Motivation
Since the appearance of the first paper on this subject in the mid-1990s, recommendation system has been growing rapidly, and flourishing in various area including music recommendations, news recommendations, web page and document recommendations. Approaches commonly proposed on recommendation system are collaborative filtering, content-based filtering, and hybrid filtering, and these approaches generate recommendation based on user data of previous ratings or reactions of users for each items. 

However, it is widely known that Collaborative filtering approach suffers from sparsity and cold start (CS) problem. This implies that if we build user-item matrix, 1) since we can collect only limited portion of user and item rating data, user-rating matrix become sparse, making it difficult to estimate relationship between user and item, and 2) it is difficult to give appropriate recommendation to new users, who does not provide any historical record or data regarding usage pattern or rating. In addition, recommendation based on user ratings also bears innate problem that if user rating does not exist it is difficult to build a recommendation model itself. The recommendation case this study tries to focus on faces the problems stated above. In case of courses for college students, 1) number of taken courses per each student is not large enough to produce recommendation and 2) course rating data for all student is not available. Therefore, it is difficult to apply existing recommendation system for college course recommendation. 

To generate recommendations for college courses and help students make decisions, this study proposes a recommender system for college courses based on topic modeling of course description, and job description of which students are interested in. The system is based on topic modeling utilizing Latent Dirichlet Allocation, and document similarity computation method using Jensen-Shannan Divergence and TF-IDF score.  Contribution of the proposed recommendation model are as the following: 1) The model generates course recommendation based on the relevance between contents of the courses, and requirement of job description each student is interested in. 2) As the database of job description is enhanced and expands, the model can be reinforced to provide better recommendation. 3) Since the model is based on descriptive textual data, the model can expand its application to any situation aims to generate intuitive recommendation without numerical user rating data. 

### Methodology
<img width="1525" alt="overview" src="https://github.com/ethHong/LDA_Job2Course_Recommender/assets/43837843/627ecd77-2f3a-4620-9022-b112fb82f7d2">
Purpose of proposed model is to recommend the most relevant document among existing document set (Course Data), when new document from outside of the dataset is given as input (Job Description). The model consists of two processes: LDA topic modeling process of course data, and distance computation process between courses and queried job description. LDA topic modeling process generates document-topic distribution matrix of course dataset. Distance computation process first query job description documents which are relevant to user input keyword, and create document-topic distribution matrix of job description using pre-trained LDA model. Then distance between course document-topic distribution and job description document-topic distribution are computed, using JS distance and TF-IDF score. This section will introduce details of each process.

### Process of document topic matrix generation
<img width="1320" alt="LDA_generation" src="https://github.com/ethHong/LDA_Job2Course_Recommender/assets/43837843/28149545-e744-4fbc-99d3-3d2ee2634ae8">

* Pre-process 682 course description data by tokenizing and cleansing. 
* Merge queried job description data into a single document, and pre-pocess by tokenizing and cleansing. 
* Generate LDA model in terms of course description data. 
* Create document-topic distribution matrix of course descriptions, denoted as C, which is a 682*40 matrix. 
* Create document-topic distribution matrix of a job description, denoted as J, which is a 1*40 vector. 
