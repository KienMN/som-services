import numpy as np
import bson

from src.config import db
from src.model_helper import parse_body_request

try:
    db.create_collection("training_data")
except Exception:
    pass


@parse_body_request
def create(dims):
    try:
        bucket = {"dims": dims, "data": []}
        col = db["training_data"]
        bucket_id = col.insert_one(bucket).inserted_id
    except Exception as err:
        return {"message": str(err)}
    else:
        return {"message": "Create success", "bucket_id": str(bucket_id)}

@parse_body_request
def push(bucket_id, data):
    try:
        arr_check = np.array(data)
        if isinstance(arr_check, np.object):
            raise ValueError("Bad data")
        doc_id = bson.ObjectId(bucket_id)
        doc = db.training_data.find_one({"_id": doc_id})
        if doc["dims"] != arr_check.shape[1]:
            raise ValueError("Bad data")
        else:
            doc["data"].entend(data)
            db.training_data.find_one_and_update({"_id": doc_id}, doc)
    except Exception as err:
        return {"message": str(err)}
    else:
        return {"message": "Push data success"}

@parse_body_request
def delete(bucket_id):
    try:
        doc_id = bson.ObjectId(bucket_id)
        db["training_data"].find_one_and_delete({"_id": doc_id})
    except Exception as err:
        return {"message": str(err)}
    else:
        return {"message": "Delete success"}
