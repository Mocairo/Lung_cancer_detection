import os
from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from datetime import datetime

db = SQLAlchemy()
login = LoginManager()
login.login_view = 'main.login'
login.login_message = "请登录以访问此页面。"
login.login_message_category = "info"

def create_app(config_class=Config):
    app = Flask(__name__,
                template_folder='../templates',
                static_folder='../static',
                instance_relative_config=True)
    app.config.from_object(config_class)

    # Manually create the instance folder if it doesn't exist
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    db.init_app(app)
    login.init_app(app)

    # This callback is used to reload the user object from the user ID stored in the session.
    @login.user_loader
    def load_user(user_id):
        from .models import User
        return User.query.get(int(user_id))

    from .routes import bp as main_bp
    app.register_blueprint(main_bp)
        
    # Register custom Jinja filter
    @app.template_filter('format_datetime')
    def format_datetime(value, format='%Y-%m-%d %H:%M'):
        from datetime import datetime
        if isinstance(value, str) and value.lower() == 'now':
            return datetime.now().strftime(format)
        if isinstance(value, datetime):
            return value.strftime(format)
        return value

    # Create database tables for our models
    with app.app_context():
        from . import models
        db.create_all()

    return app 