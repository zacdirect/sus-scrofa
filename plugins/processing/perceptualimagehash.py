# SusScrofa - Copyright (C) 2013-2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

"""
DEPRECATED: This processing module has been superseded by
    plugins/analyzer/perceptual_hash.py

The new analyzer module adds:
  - Hamming distance scoring (not just exact-match)
  - Wavelet hash (wHash)
  - Cross-case similar image search
  - Perceptual hash list matching
  - Inter-hash manipulation fingerprinting
  - Confidence scoring integration

This module remains for backward compatibility with existing results
that reference the 'imghash' and 'similar' keys. New installations
should rely on the analyzer module (auto-discovered).
"""

import warnings

from analyses.models import Case
from lib.analyzer.base import BaseProcessingModule

from lib.utils import str2image

warnings.warn(
    "perceptualimagehash processing module is deprecated. "
    "Use plugins/analyzer/perceptual_hash.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    import imagehash
    IS_IMAGEHASH = True
except ImportError:
    IS_IMAGEHASH = False


class PerceptualImageHashProcessing(BaseProcessingModule):
    """Perceptual Image Hashing.
    
    DEPRECATED: Superseded by plugins/analyzer/perceptual_hash.PerceptualHashAnalyzer
    """

    name = "ImageHash Calculator (Legacy)"
    description = "Legacy perceptual hashing — use analyzer/perceptual_hash instead."
    order = 10

    # Disabled by default — the new analyzer module handles this
    enabled = False

    HASH_SIZE = 8

    def check_deps(self):
        # Return False to prevent this module from running
        # The new PerceptualHashAnalyzer (plugins/analyzer/perceptual_hash.py) handles this
        return False

    def get_similar_images(self, hash_value, hash_func):
        # TODO: this should be refactored in the future.

        # Map.
        if hash_func == imagehash.average_hash:
            hash_name = "a_hash"
        elif hash_func == imagehash.phash:
            hash_name = "p_hash"
        elif hash_func == imagehash.dhash:
            hash_name = "d_hash"

        # Search.
        image_hash = imagehash.hex_to_hash(hash_value)
        similarities = list()
        for img in self.task.case.images.filter(state="C").exclude(id=self.task.id):
            if img.report and \
            "imghash" in img.report and \
            hash_name in img.report["imghash"] and \
            image_hash == imagehash.hex_to_hash(img.report["imghash"][hash_name]):
                # TODO: store also image distance.
                similarities.append(img.id)
        return similarities


    def run(self, task):
        self.task = task
        image = str2image(task.get_file_data)

        # Calculate hash.
        self.results["imghash"]["a_hash"] = str(imagehash.average_hash(image, hash_size=self.HASH_SIZE))
        self.results["imghash"]["p_hash"] = str(imagehash.phash(image, hash_size=self.HASH_SIZE))
        self.results["imghash"]["d_hash"] = str(imagehash.dhash(image, hash_size=self.HASH_SIZE))

        # Get similar images.
        self.results["similar"]["a_hash"] = self.get_similar_images(self.results["imghash"]["a_hash"], imagehash.average_hash)
        self.results["similar"]["p_hash"] = self.get_similar_images(self.results["imghash"]["p_hash"], imagehash.phash)
        self.results["similar"]["d_hash"] = self.get_similar_images(self.results["imghash"]["d_hash"], imagehash.dhash)

        return self.results
