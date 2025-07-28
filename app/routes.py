from flask import Blueprint

bp = Blueprint('main', __name__)
 
# Import other routes here if needed
from . import views, auth 