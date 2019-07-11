# bad
import sys
import os
src_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(src_path)

import connexion
from flask_cors import CORS


def create_app():
    app = connexion.App(__name__, specification_dir="./specs")
    app.add_api("openapi.yaml")
    CORS(app.app)

    return app

