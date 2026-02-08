# SusScrofa - Copyright (C) 2013-2026 SusScrofa Developers.
# This file is part of SusScrofa.
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
            
            # Store all results (including new verdict fields)
            self.results["confidence"]["manipulation_detected"] = confidence['manipulation_detected']
            self.results["confidence"]["confidence_score"] = confidence['confidence_score']
            self.results["confidence"]["ai_generated_probability"] = confidence['ai_generated_probability']
            self.results["confidence"]["indicators"] = confidence['indicators']
            self.results["confidence"]["methods"] = confidence['methods']
            
            # Store verdict fields
            self.results["confidence"]["verdict"] = confidence['verdict']
            self.results["confidence"]["verdict_label"] = confidence['verdict_label']
            self.results["confidence"]["verdict_confidence"] = confidence['verdict_confidence']
            self.results["confidence"]["verdict_certainty"] = confidence['verdict_certainty']
            
            # Store method breakdown
            self.results["confidence"]["deterministic_methods"] = confidence.get('deterministic_methods', {})
            self.results["confidence"]["ai_ml_methods"] = confidence.get('ai_ml_methods', {})
            
            logger.info(f"[Task {task.id}]: Confidence scoring complete - "
                       f"Manipulation: {confidence['confidence_score']:.2f}%, "
                       f"AI: {confidence['ai_generated_probability']:.2f}%")
            
        except Exception as e:
            logger.exception(f"[Task {task.id}]: Error in confidence scoring: {e}")
        
        return self.results
