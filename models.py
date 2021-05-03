from extensions import db
#유저가 인풋하는 모델

class User_rating(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    user = db.Column(db.Text)
    keyword = db.Column(db.Text)
    major = db.Column(db.Text)
    rating = db.Column(db.Integer)
    question = db.Column(db.Text)
    division = db.Column(db.Text)
    recommended_course = db.Column(db.Text)
    engine = db.Column(db.Text)
    interview = db.Column(db.Text)

    def __init__(self, user, keyword, major, rating, division, recommended_course, engine, question, interview):
        self.user = user
        self.keyword = keyword
        self.major = major
        self.rating = rating
        self.question = question
        self.interview = interview
        self.division = division
        self.recommended_course = recommended_course
        self.engine = engine
