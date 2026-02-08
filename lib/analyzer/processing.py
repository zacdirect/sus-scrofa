# SusScrofa - Copyright (C) 2026 SusScrofa Developers.
# This file is part of SusScrofa.
# See the file 'docs/LICENSE.txt' for license terms.

import logging

from time import sleep
from multiprocessing import cpu_count, Process, JoinableQueue
from django.utils.timezone import now
from django.conf import settings

from lib.analyzer.orchestrator import (
    discover_plugins, check_deps, should_skip_ai_ml, run_plugins,
)
from lib.analyzer.auditor import audit
from lib.forensics.confidence import calculate_manipulation_confidence
from lib.db import save_results, update_results
from analyses.models import Analysis

logger = logging.getLogger(__name__)


class AnalysisRunner(Process):
    """Run an analysis process."""

    def __init__(self, tasks, static_plugins=None, ai_ml_plugins=None):
        Process.__init__(self)
        self.tasks = tasks
        self.static_plugins = static_plugins or []
        self.ai_ml_plugins = ai_ml_plugins or []
        logger.debug("AnalysisRunner started")

    def run(self):
        """Start processing."""
        while True:
            try:
                task = self.tasks.get()
                self._process_image(task)
            except KeyboardInterrupt:
                break

    def _process_image(self, task):
        """Process an image.

        Three-phase architecture:

          Phase 1a — Static plugins:
                     Fast, deterministic analysers (metadata, hashes,
                     ELA, noise, frequency, signatures, …).

          ── Auditor checkpoint ──
                     If static evidence alone is already decisive,
                     skip the expensive AI/ML phase.

          Phase 1b — AI/ML plugins:
                     Heavier detectors (SDXL, SPAI, OpenCV manipulation,
                     photoholmes, …).  Only run when Phase 1a didn't
                     already produce a clear verdict.

          Phase 2  — Engine post-processing (NOT plugins):
                     a) Compliance Auditor → results['audit']
                     b) Confidence Scoring → results['confidence']

        @param task: image task
        """
        try:
            results = {}
            results["file_data"] = task.image_id

            # ── Phase 1a: Static plugins ──────────────────────────
            run_plugins(self.static_plugins, results, task)

            # ── Auditor checkpoint ────────────────────────────────
            skipped_ai = False
            if self.ai_ml_plugins and should_skip_ai_ml(results):
                skipped_ai = True
                results["_engine"] = {"ai_ml_skipped": True,
                                      "skip_reason": "decisive static evidence"}
                logger.info("Task %s: AI/ML tier skipped (static evidence sufficient)", task.id)
            else:
                # ── Phase 1b: AI/ML plugins ───────────────────────
                run_plugins(self.ai_ml_plugins, results, task)

            # ── Phase 2: Engine post-processing ───────────────────
            try:
                results["audit"] = audit(results)
                logger.info("Compliance audit complete for task %s: authenticity=%s",
                            task.id, results["audit"].get("authenticity_score"))
            except Exception as e:
                logger.exception("Compliance audit failed for task %s: %s", task.id, e)
                results["audit"] = {"error": str(e)}

            try:
                results["confidence"] = calculate_manipulation_confidence(results)
            except Exception as e:
                logger.exception("Confidence scoring failed for task %s: %s", task.id, e)
                results["confidence"] = {"error": str(e)}

            # Complete — save or update results.
            # If analysis_id exists, we're re-processing → update existing MongoDB doc.
            if task.analysis_id:
                update_results(task.analysis_id, results)
                logger.info("Re-processed task {0} with success (updated existing analysis)".format(task.id))
            else:
                task.analysis_id = save_results(results)
                logger.info("Processed task {0} with success (new analysis)".format(task.id))
            
            task.state = "C"
        except Exception as e:
            logger.exception("Critical error processing task {0}, skipping task: {1}".format(task.id, e))
            task.state = "F"
        finally:
            # Save.
            task.completed_at = now()
            task.save()
            self.tasks.task_done()


class AnalysisManager():
    """Manage all analysis' process."""

    def __init__(self):
        logger.debug("Using pool on %i core" % self.get_parallelism())
        # Discover and validate plugins via engine orchestrator.
        static_raw, ai_ml_raw = discover_plugins()
        self.static_plugins = check_deps(static_raw)
        self.ai_ml_plugins = check_deps(ai_ml_raw)
        logger.info("Loaded %d static plugins, %d AI/ML plugins",
                     len(self.static_plugins), len(self.ai_ml_plugins))
        # Starting worker pool.
        self.workers = []
        self.tasks = JoinableQueue(self.get_parallelism())
        self.workers_start()

    def workers_start(self):
        """Start workers pool."""
        for _ in range(self.get_parallelism()):
            runner = AnalysisRunner(self.tasks, self.static_plugins, self.ai_ml_plugins)
            runner.start()
            self.workers.append(runner)

    def workers_stop(self):
        """Stop workers pool."""
        # Wait for.
        for sex_worker in self.workers:
            sex_worker.join()

    def get_parallelism(self):
        """Get the sus_scrofa parallelism level for analysis processing."""
        # Check database type. If we detect SQLite we slow down processing to
        # only one process. SQLite does not support parallelism.
        if settings.DATABASES["default"]["ENGINE"].endswith("sqlite3"):
            logger.warning("Detected SQLite database, decreased parallelism to 1. SQLite doesn't support parallelism.")
            return 1
        elif cpu_count() > 1:
            # Set it to total CPU minus one let or db and other use.
            return cpu_count() - 1
        else:
            return 1

    def run(self):
        """Start all analyses."""
        # Clean up tasks remaining stale from old runs.
        if Analysis.objects.filter(state="P").exists():
            logger.info("Found %i stale analysis, putting them in queue." % Analysis.objects.filter(state="P").count())
            Analysis.objects.filter(state="P").update(state="W")

        # Infinite finite loop.
        try:
            while True:
                # Fetch tasks waiting processing.
                tasks = Analysis.objects.filter(state="W").order_by("id")

                if tasks.exists() and not self.tasks.full():
                    # Using iterator() to avoid caching.
                    for task in Analysis.objects.filter(state="W").order_by("id").iterator():
                        self.tasks.put(task)
                        logger.debug("Processing task %s" % task.id)
                        task.state = "P"
                        task.save()
                elif self.tasks.full():
                    logger.debug("Queue full. Waiting...")
                    sleep(1)
                else:
                    logger.debug("No tasks. Waiting...")
                    sleep(1)
        except KeyboardInterrupt:
            print("Exiting... (requested by user)")
        finally:
            print("Waiting tasks to accomplish...")
            self.workers_stop()
            print("Processing done. Have a nice day in the real world.")
