import os
import sys
import gridfs

from django.core.management.base import BaseCommand
from bson.objectid import InvalidId

from analyses.models import Analysis
from lib.db import get_file, mongo_connect


class Command(BaseCommand):

    help = "Save all images to a directory"

    def add_arguments(self, parser):
        parser.add_argument("-p", "--path", type=str, help="Path to save images")

    def handle(self, *args, **options):
        print("Starting")

        if os.path.exists(options["path"]):
            dst_path = os.path.join(options["path"], "sus_scrofa_output")
            if os.path.exists(dst_path):
                print("ERROR: a folder 'sus_scrofa_output' already exist in that path!")
                sys.exit()
            else:
                # Create destination folder.
                os.mkdir(dst_path)
                # Mongo connection.
                db = mongo_connect()
                fs = gridfs.GridFS(db)
                # We are fine, run!
                for analysis in Analysis.objects.all():
                    try:
                       file = get_file(analysis.image_id)
                    except (InvalidId, TypeError) as e:
                        print("Unable to dump %s: %s" % (analysis.id, e))
                        continue
                    else:
                        with open(os.path.join(dst_path, "analysis_%s" % analysis.id), "a") as the_file:
                            the_file.write(file.read())
        else:
            print("ERROR: path not found!")