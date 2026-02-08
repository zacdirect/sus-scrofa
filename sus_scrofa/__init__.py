# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

from pymongo import GEO2D

# Mongo connection - Lazy loading to avoid circular imports
# The MongoDB connection and index creation will be done on first access
# from lib.db import mongo_connect
# db = mongo_connect()

# Indexes - will be created when MongoDB is accessed
# db.fs.files.ensure_index("sha1", unique=True, name="sha1_unique")
# db.fs.files.ensure_index("uuid", unique=True, name="uuid_unique")
# db.analyses.ensure_index([("metadata.gps.pos", GEO2D)])