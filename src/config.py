import os
import logging
import pymongo

# Synchronous with gunicorn loggger
logger = logging.getLogger('gunicorn.error')

try:
    conn = pymongo.MongoClient(os.environ["DB_HOST"], int(os.environ["DB_PORT"]))
    db = conn.get_database(os.environ["DB_NAME"])
    if "DB_USER" in os.environ and "DB_PASS" in os.environ:
        db.authenticate(os.environ["DB_USER"], os.environ["DB_PASS"])
except Exception:
    pass

model_dir = os.environ["MODEL_DIR"]

