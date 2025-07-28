from flask_login import UserMixin
from . import db
from datetime import datetime
from sqlalchemy.types import JSON

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    screenings = db.relationship('ScreeningSession', backref='user', lazy='dynamic')

class ScreeningSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    
    # Questionnaire data
    age = db.Column(db.Integer)
    gender = db.Column(db.String(50))
    smoking_status = db.Column(db.String(50))
    packs_per_day = db.Column(db.Integer, default=0)
    years_smoked = db.Column(db.Integer, default=0)
    pack_years = db.Column(db.Integer)
    family_history = db.Column(db.String(10))
    occupational_exposure = db.Column(db.String(10))
    previous_cancer_history = db.Column(db.String(10))

    # File data
    original_filename = db.Column(db.String(256))
    saved_filename = db.Column(db.String(256)) # The secure filename saved on the server

    # Analysis results
    analysis_result = db.Column(JSON)
    status = db.Column(db.String(50), default='Pending Analysis') # e.g., Pending, Processing, Complete, Failed
    ai_summary = db.Column(db.Text)

    # Paths to generated visualization images
    image_2d_slice_top = db.Column(db.String(256))
    image_2d_slice_center = db.Column(db.String(256))
    image_2d_slice_bottom = db.Column(db.String(256))
    image_3d_default = db.Column(db.String(256))
    image_3d_mip = db.Column(db.String(256))
    image_3d_iso = db.Column(db.String(256))

    def __repr__(self):
        return f'<ScreeningSession {self.id} for User {self.user_id}>' 