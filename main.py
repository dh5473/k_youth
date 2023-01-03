from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from models import model

from common import config

db = SQLAlchemy()
migrate = Migrate()


def create_app():
    app = Flask(__name__)
    app.config.from_object(config)

    # ORM
    db_ = model.db
    db_.init_app(app)
    migrate.init_app(app, db_)
    db_.app = app
    db_.create_all()

    # blueprint
    from views import index_views, gradcam_views
    app.register_blueprint(index_views.bp)
    app.register_blueprint(gradcam_views.bp)

    return app