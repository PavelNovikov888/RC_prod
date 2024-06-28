from app import db

class Flat(db.Model):
    id = db.Column(db.Integer, primary_key=True, unique=True)
    cost = db.Column(db.Integer)

