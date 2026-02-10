# Sus Scrofa - Copyright (C) 2026 Sus Scrofa Developers.
# This file is part of Sus Scrofa.
# See the file 'docs/LICENSE.txt' for license terms.

from django.apps import AppConfig
from pymongo import GEO2D

# Mongo connection.
from lib.db import mongo_connect


class SystemConfig(AppConfig):
    name = "system"
    verbose_name = "System Application"

    def ready(self):
        """Initialization code.
        It runs once at app startup.
        """
        db = mongo_connect()
        
        if db is None:
            return

        # Create indexes individually, continuing even if some fail
        # GridFS file indexes
        try:
            db.fs.files.create_index("sha1", unique=True, name="sha1_unique")
        except Exception as e:
            # Index might already exist or have duplicate keys, that's okay
            pass
            
        try:
            db.fs.files.create_index("uuid", unique=True, name="uuid_unique")
        except Exception as e:
            # Index might already exist or have duplicate keys, that's okay
            pass
        
        # Geospatial index for GPS searches - this is critical
        try:
            db.analyses.create_index([("metadata.gps.pos", GEO2D)])
            print("✓ Geospatial index created successfully")
        except Exception as e:
            print(f"Warning: Could not create geospatial index: {e}")
