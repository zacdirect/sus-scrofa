# Ghiro - Copyright (C) 2013-2015 Ghiro Developers.
# This file is part of Ghiro.
# See the file 'docs/LICENSE.txt' for license terms.

import logging
from django.core.management.base import BaseCommand

from lib.analyzer.processing import AnalysisManager

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    """Process images on analysis queue."""

    help = "Image processing"

    def handle(self, *args, **options):
        """Runs command."""
        logger.debug("Starting processor...")

        try:
            AnalysisManager().run()
        except KeyboardInterrupt:
            print("Exiting... (requested by user)")
