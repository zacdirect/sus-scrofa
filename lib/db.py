# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

import sys
import hashlib
import uuid
from django.conf import settings
import gridfs
from bson import ObjectId
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure

# Lazy MongoDB connection
_db = None
_fs = None
_mongo_available = None

def mongo_connect():
    """Connects to Mongo, returns None if unable to connect.
    @return: connection handler or None
    """
    global _mongo_available
    try:
        db = Database(MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=2000), settings.MONGO_DB)
        # Test connection
        db.client.server_info()
        _mongo_available = True
        return db
    except (ConnectionFailure, Exception) as e:
        print(f"WARNING: Unable to connect to MongoDB: {e}")
        print("MongoDB features will be unavailable. Install and start MongoDB to use image analysis features.")
        _mongo_available = False
        return None

def get_db():
    """Get or create MongoDB connection."""
    global _db
    if _db is None:
        _db = mongo_connect()
    return _db

def get_fs():
    """Get or create GridFS connection."""
    global _fs
    if _fs is None:
        db = get_db()
        if db is not None:
            _fs = gridfs.GridFS(db)
    return _fs

def is_mongo_available():
    """Check if MongoDB is available."""
    if _mongo_available is None:
        get_db()
    return _mongo_available

# For backward compatibility - these are now just aliases
db = get_db()
fs = get_fs()

def save_file(data=None, file_path=None, content_type=None):
    """Save file in GridFS.
    @param data: file data
    @param file_path: file path
    @param content_type: file content type
    @return: saved file ID
    """
    fs = get_fs()
    db = get_db()
    
    if file_path:
        try:
            fh = open(file_path, "rb")
            data = fh.read()
        finally:
            fh.close()

    # Using SHA1 as file key.
    sha1 = hashlib.sha1(data).hexdigest()

    # File identifier.
    id = uuid.uuid4().hex + sha1

    # Save file and returns UUID.
    try:
        fs.put(data, content_type=content_type, sha1=sha1, uuid=id)
    except gridfs.errors.FileExists:
        id = db.fs.files.find_one({"sha1": sha1})["uuid"]
    finally:
        return id

def get_file(id):
    """Get a file from GridFS.
    @param id: file uuid
    @return: file object"""
    fs = get_fs()
    db = get_db()
    obj_id = db.fs.files.find_one({"uuid": id})["_id"]
    return fs.get(ObjectId(obj_id))

def get_file_length(id):
    """Get a file lenght from GridFS.
    @param id: file uuid
    @return: integer"""
    db = get_db()
    return db.fs.files.find_one({"uuid": id})["length"]

def save_results(results):
    """Save results in mongo.
    @param results: data dict
    @return: object id
    """
    db = get_db()
    result = db.analyses.insert_one(results)
    return result.inserted_id

def update_results(analysis_id, results):
    """Update existing results in mongo (for re-processing).
    @param analysis_id: existing MongoDB ObjectId (as string)
    @param results: new data dict to replace
    """
    db = get_db()
    db.analyses.replace_one(
        {"_id": ObjectId(analysis_id)},
        results
    )