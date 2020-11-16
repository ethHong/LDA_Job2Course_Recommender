# -Course-Recommentation-Project

## For Details, please look at README of following projects:https://github.com/ethHong/LDA_course_recommender

## How it works
Using LDA and Jensen Shannen Divergence, this projects tries to conpute disatnce between each courses job description of user's input

![Current](https://user-images.githubusercontent.com/43837843/99258349-c78d6700-285b-11eb-8b48-203647830d03.png)



## Desctipion
* Use program_main.py to run the program. The program initialize by asking ID, PASSWIRD if linkeding for JD access
* Put Industry, Career position (job title) and company of interest
* The program will proceed collecting some of recent relevant JDs, and let you choose top 3 you prefer
* Based on JDs you choose, the model will recommend top 10 courses from Yonsei University UIC Course (2015 ~ 2020)

## Improvements and more works
### This project has following limitation, which the author is working to improve
* Problem of Crawler's unstableness
* Range of Job Descriptions searched by keyword is too broad
* Looking for ways to recommend 'portfolio' of classed, by giving weights to three factors as different dimension: Company, Industry, and Job title
* To improve user experience: trying to work on building DB, and add metadata as tag
![Update](https://user-images.githubusercontent.com/43837843/99258418-e3910880-285b-11eb-9f51-af74bf972020.png)
