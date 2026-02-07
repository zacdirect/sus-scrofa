# Ghiro - Copyright (C) 2013-2026 Ghiro Developers.
# This file is part of Ghiro.
# See the file 'docs/LICENSE.txt' for license terms.

import logging

from lib.analyzer.base import BaseAnalyzerModule
from lib.forensics.confidence import calculate_manipulation_confidence

logger = logging.getLogger(__name__)


class ConfidenceScoringProcessing(BaseAnalyzerModule):
    """Aggregates all detection results into confidence scores."""

    name = "Confidence Scoring"
    description = "Calculates overall manipulation and AI generation confidence scores."
    order = 90  # Run last to aggregate all results

    def check_deps(self):
        return True  # No special dependencies

    def run(self, task):
        try:
            # Calculate confidence based on all accumulated results
            confidence = calculate_manipulation_confidence(self.data)
            
            # Store results
            self.results["confidence"]["manipulation_detected"] = confidence['manipulation_detected']
            self.results["confidence"]["confidence_score"] = confidence['confidence_score']
            self.results["confidence"]["ai_generated_probability"] = confidence['ai_generated_probability']
            self.results["confidence"]["indicators"] = confidence['indicators']
            self.results["confidence"]["methods"] = confidence['methods']
            
            logger.info(f"[Task {task.id}]: Confidence scoring complete - "
                       f"Manipulation: {confidence['confidence_score']:.2%}, "
                       f"AI: {confidence['ai_generated_probability']:.2%}")
            
        except Exception as e:
            logger.exception(f"[Task {task.id}]: Error in confidence scoring: {e}")
        
        return self.results
