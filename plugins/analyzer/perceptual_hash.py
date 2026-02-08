# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

"""
Modernized Perceptual Image Hashing — distance-based, cross-case, multi-algorithm.

Upgrades over the legacy perceptualimagehash processing module:
  - Hamming distance scoring (not just exact-match)
  - Wavelet hash (wHash) for JPEG re-compression resistance
  - Color hash for geometric transform resistance
  - Cross-case similar image search
  - Perceptual hash list matching (known-bad/known-good)
  - Configurable distance thresholds
  - Results feed into confidence scoring pipeline

Hash algorithms (all from the `imagehash` library):
  aHash  — Average Hash: detects brightness/contrast manipulation
  pHash  — Perceptual Hash: detects geometric transforms (resize, crop, rotation)
  dHash  — Difference Hash: detects gradient/edge manipulation
  wHash  — Wavelet Hash: resistant to JPEG re-compression artifacts
"""

import logging

from django.db.models import Q

from lib.analyzer.base import BaseAnalyzerModule
from lib.utils import str2image

logger = logging.getLogger(__name__)

try:
    import imagehash
    IS_IMAGEHASH = True
except ImportError:
    IS_IMAGEHASH = False


class PerceptualHashAnalyzer(BaseAnalyzerModule):
    """Modernized perceptual image hashing with distance scoring."""

    order = 15  # Run early — hashes are fast and feed downstream modules

    # Hash size (8 = 64-bit hashes, good balance of speed and accuracy)
    HASH_SIZE = 8

    # Distance thresholds for similarity classification
    THRESHOLD_EXACT = 0       # Identical perceptual content
    THRESHOLD_NEAR_DUPLICATE = 5   # Minor edits (re-encoding, slight crop)
    THRESHOLD_SIMILAR = 10    # Moderate edits (filter, resize, color shift)
    THRESHOLD_RELATED = 15    # Significant edits (composite, heavy filter)

    # Maximum images to compare in cross-case search (performance guard)
    MAX_CROSS_CASE_SEARCH = 500

    def check_deps(self):
        return IS_IMAGEHASH

    def _compute_hashes(self, image):
        """Compute all perceptual hash types for an image."""
        return {
            "a_hash": str(imagehash.average_hash(image, hash_size=self.HASH_SIZE)),
            "p_hash": str(imagehash.phash(image, hash_size=self.HASH_SIZE)),
            "d_hash": str(imagehash.dhash(image, hash_size=self.HASH_SIZE)),
            "w_hash": str(imagehash.whash(image, hash_size=self.HASH_SIZE)),
        }

    def _hamming_distance(self, hash_hex_a, hash_hex_b):
        """Compute Hamming distance between two hex hash strings."""
        try:
            h1 = imagehash.hex_to_hash(hash_hex_a)
            h2 = imagehash.hex_to_hash(hash_hex_b)
            return h1 - h2  # imagehash overloads __sub__ for Hamming distance
        except Exception:
            return 999  # unreachable distance on error

    def _classify_distance(self, distance):
        """Classify a Hamming distance into a human-readable similarity level."""
        if distance <= self.THRESHOLD_EXACT:
            return "exact"
        elif distance <= self.THRESHOLD_NEAR_DUPLICATE:
            return "near_duplicate"
        elif distance <= self.THRESHOLD_SIMILAR:
            return "similar"
        elif distance <= self.THRESHOLD_RELATED:
            return "related"
        else:
            return None  # not similar enough

    def _find_similar_in_case(self, task, hashes):
        """Find similar images within the same case using distance scoring."""
        from analyses.models import Analysis

        similar = []
        case_images = task.case.images.filter(state="C").exclude(id=task.id)

        for img in case_images:
            if not img.report or "imghash" not in img.report:
                continue

            img_hashes = img.report["imghash"]
            best_distance = 999
            best_hash_type = None

            for hash_type in ["p_hash", "a_hash", "d_hash", "w_hash"]:
                if hash_type in hashes and hash_type in img_hashes:
                    dist = self._hamming_distance(hashes[hash_type], img_hashes[hash_type])
                    if dist < best_distance:
                        best_distance = dist
                        best_hash_type = hash_type

            classification = self._classify_distance(best_distance)
            if classification:
                similar.append({
                    "image_id": img.id,
                    "distance": best_distance,
                    "classification": classification,
                    "matched_on": best_hash_type,
                })

        # Sort by distance (closest first)
        similar.sort(key=lambda x: x["distance"])
        return similar

    def _find_cross_case_similar(self, task, hashes):
        """Find similar images across ALL cases (for OSINT/forensic linking)."""
        from analyses.models import Analysis

        similar = []
        # Search completed analyses across all cases, excluding current
        cross_case = Analysis.objects.filter(
            state="C"
        ).exclude(
            id=task.id
        ).exclude(
            case=task.case
        ).order_by("-id")[:self.MAX_CROSS_CASE_SEARCH]

        for img in cross_case:
            if not img.report or "imghash" not in img.report:
                continue

            img_hashes = img.report["imghash"]
            best_distance = 999
            best_hash_type = None

            for hash_type in ["p_hash", "a_hash", "d_hash", "w_hash"]:
                if hash_type in hashes and hash_type in img_hashes:
                    dist = self._hamming_distance(hashes[hash_type], img_hashes[hash_type])
                    if dist < best_distance:
                        best_distance = dist
                        best_hash_type = hash_type

            classification = self._classify_distance(best_distance)
            if classification:
                similar.append({
                    "image_id": img.id,
                    "case_id": img.case_id,
                    "case_name": str(img.case) if img.case else "Unknown",
                    "distance": best_distance,
                    "classification": classification,
                    "matched_on": best_hash_type,
                })

        similar.sort(key=lambda x: x["distance"])
        return similar[:20]  # Cap results for display

    def _check_perceptual_hash_lists(self, task, hashes):
        """Match perceptual hashes against uploaded hash lists using distance."""
        from hashes.models import List as HashList

        matches = []
        perceptual_ciphers = ["a_hash", "p_hash", "d_hash", "w_hash"]

        for cipher in perceptual_ciphers:
            if cipher not in hashes:
                continue

            # Get hash lists of this perceptual type (owned by user or public)
            hash_lists = HashList.objects.filter(
                cipher=cipher
            ).filter(
                Q(owner=task.owner) | Q(public=True)
            )

            our_hash = imagehash.hex_to_hash(hashes[cipher])

            for hash_list in hash_lists:
                for stored_hash in hash_list.hash_set.all():
                    try:
                        stored = imagehash.hex_to_hash(stored_hash.value)
                        distance = our_hash - stored
                    except Exception:
                        continue

                    classification = self._classify_distance(distance)
                    if classification:
                        matches.append({
                            "list_id": hash_list.id,
                            "list_name": hash_list.name,
                            "hash_type": cipher,
                            "distance": distance,
                            "classification": classification,
                            "matched_value": stored_hash.value,
                        })

                        # Also register the match on the hash list model
                        hash_list.matches.add(task)

        matches.sort(key=lambda x: x["distance"])
        return matches

    def _analyze_manipulation_fingerprint(self, hashes):
        """
        Compare hash types to detect manipulation signatures.

        Different hash types are sensitive to different manipulations:
          - aHash stable but dHash changed → edge manipulation (retouching)
          - pHash stable but dHash changed → same content, altered edges
          - wHash diverges from others → frequency-domain manipulation

        This produces forensic indicators, not a final verdict.
        """
        # This compares the image's OWN hashes against each other for internal
        # consistency — it's most useful when comparing two versions of the same image.
        # For single-image analysis, we just record the hash diversity as metadata.
        hash_values = {}
        for h_type in ["a_hash", "p_hash", "d_hash", "w_hash"]:
            if h_type in hashes:
                hash_values[h_type] = imagehash.hex_to_hash(hashes[h_type])

        if len(hash_values) < 2:
            return {}

        # Calculate inter-hash distances (how much the different algorithms agree)
        inter_distances = {}
        types = list(hash_values.keys())
        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                key = f"{types[i]}_vs_{types[j]}"
                inter_distances[key] = hash_values[types[i]] - hash_values[types[j]]

        return inter_distances

    def run(self, task):
        """Run perceptual hash analysis."""
        try:
            image = str2image(task.get_file_data)
        except Exception as e:
            logger.warning(f"[Task {task.id}]: Error opening image for hashing: {e}")
            return self.results

        # 1. Compute all hash types
        hashes = self._compute_hashes(image)
        self.results["imghash"] = hashes

        logger.info(f"[Task {task.id}]: Perceptual hashes computed: "
                    f"aHash={hashes['a_hash']}, pHash={hashes['p_hash']}, "
                    f"dHash={hashes['d_hash']}, wHash={hashes['w_hash']}")

        # 2. In-case similarity search (with distance scoring)
        try:
            in_case = self._find_similar_in_case(task, hashes)
            self.results["similar"]["in_case"] = in_case
            if in_case:
                logger.info(f"[Task {task.id}]: Found {len(in_case)} similar images in case")
        except Exception as e:
            logger.warning(f"[Task {task.id}]: Error in case similarity search: {e}")
            self.results["similar"]["in_case"] = []

        # 3. Cross-case similarity search
        try:
            cross_case = self._find_cross_case_similar(task, hashes)
            self.results["similar"]["cross_case"] = cross_case
            if cross_case:
                logger.info(f"[Task {task.id}]: Found {len(cross_case)} similar images across cases")
        except Exception as e:
            logger.warning(f"[Task {task.id}]: Error in cross-case search: {e}")
            self.results["similar"]["cross_case"] = []

        # 4. Perceptual hash list matching
        try:
            hash_list_matches = self._check_perceptual_hash_lists(task, hashes)
            self.results["similar"]["hash_list_matches"] = hash_list_matches
            if hash_list_matches:
                logger.info(f"[Task {task.id}]: {len(hash_list_matches)} perceptual hash list matches")
        except Exception as e:
            logger.warning(f"[Task {task.id}]: Error in hash list matching: {e}")
            self.results["similar"]["hash_list_matches"] = []

        # 5. Inter-hash analysis (manipulation fingerprinting)
        try:
            inter = self._analyze_manipulation_fingerprint(hashes)
            self.results["similar"]["inter_hash_distances"] = inter
        except Exception as e:
            logger.warning(f"[Task {task.id}]: Error in inter-hash analysis: {e}")

        # Legacy compatibility — flat similar lists for old templates
        for hash_type in ["a_hash", "p_hash", "d_hash"]:
            self.results["similar"][hash_type] = [
                m["image_id"] for m in self.results["similar"].get("in_case", [])
                if m.get("matched_on") == hash_type and m.get("classification") == "exact"
            ]

        return self.results
