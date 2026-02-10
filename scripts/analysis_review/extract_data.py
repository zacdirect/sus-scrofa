#!/usr/bin/env python
"""
Extract analysis data for specific images to review detection algorithm.

Usage:
    python extract_data.py               # extract all
    python extract_data.py 10            # extract image 10 only
    python extract_data.py 8 9 10        # extract specific images
"""

import json
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sus_scrofa.settings')
django.setup()

from analyses.models import Analysis
from lib.db import get_db


def extract_analysis_data(analysis_id):
    """Extract all relevant data for an analysis.

    Pulls from both SQLite (Analysis model) and MongoDB (report document).
    Structured around the zero-trust auditor model:
        - audit_metadata is THE source of truth
        - individual method results are supporting evidence only
    """
    try:
        analysis = Analysis.objects.get(id=analysis_id)
        report = analysis.report

        if not report:
            return {"error": f"No report found for analysis {analysis_id}"}

        exif_data = report.get("metadata", {}).get("Exif", {})
        xmp_data = report.get("metadata", {}).get("XMP", {})
        ai_det = report.get("ai_detection", {})
        conf = report.get("confidence", {})

        data = {
            "id": analysis_id,
            "file_name": analysis.file_name,
            "state": analysis.state,
            "completed_at": str(analysis.completed_at) if analysis.completed_at else None,

            # ── Metadata summary ──────────────────────────────────
            "metadata": {
                "has_exif": bool(exif_data),
                "has_xmp": bool(xmp_data),
                "has_gps": bool(report.get("metadata", {}).get("gps")),
                "dimensions": report.get("metadata", {}).get("dimensions"),
                "pillow_info": report.get("metadata", {}).get("pillow_info"),
            },

            # ── EXIF key fields ───────────────────────────────────
            "exif_summary": _extract_exif_summary(exif_data),

            # ── Software / generator signatures ───────────────────
            "software_info": _extract_software_info(exif_data, xmp_data),

            # ── Zero-Trust Auditor Verdict (PRIMARY) ──────────────
            "auditor": {
                "authenticity_score": conf.get("authenticity_score"),
                "verdict": conf.get("verdict"),
                "verdict_label": conf.get("verdict_label"),
                "verdict_confidence": conf.get("verdict_confidence"),
                "verdict_certainty": conf.get("verdict_certainty"),
                "detected_types": conf.get("detected_types", []),
                "manipulation_detected": conf.get("manipulation_detected"),
            },

            # ── Audit metadata (from ComplianceAuditor) ──────────
            "audit_metadata": ai_det.get("audit_metadata", {}),

            # ── AI Detection (full) ───────────────────────────────
            "ai_detection": {
                "enabled": ai_det.get("enabled"),
                "verdict": ai_det.get("verdict"),
                "confidence": ai_det.get("confidence"),
                "likely_ai": ai_det.get("likely_ai"),
                "ai_probability": ai_det.get("ai_probability"),
                "authenticity_score": ai_det.get("authenticity_score"),
                "evidence": ai_det.get("evidence"),
                "interpretation": ai_det.get("interpretation"),
                "detection_framework": ai_det.get("detection_framework"),
                "available_methods": ai_det.get("available_methods", []),
                "detection_layers": ai_det.get("detection_layers", []),
            },

            # ── Confidence (display-only indicators) ──────────────
            "confidence_indicators": {
                "confidence_score": conf.get("confidence_score"),
                "ai_generated_probability": conf.get("ai_generated_probability"),
                "indicators": conf.get("indicators", []),
                "deterministic_methods": conf.get("deterministic_methods", {}),
                "ai_ml_methods": conf.get("ai_ml_methods", {}),
            },

            # ── Forensic methods (supporting evidence) ────────────
            "ela": {
                "max_difference": report.get("ela", {}).get("max_difference"),
                "has_image": bool(report.get("ela", {}).get("ela_image")),
            },
            "noise_analysis": report.get("noise_analysis", {}),
            "frequency_analysis": report.get("frequency_analysis", {}),

            # ── OpenCV (computer vision service) ──────────────────
            "opencv_analysis": report.get("opencv_analysis", {}),
            "opencv_manipulation": report.get("opencv_manipulation", {}),

            # ── Perceptual hashes ─────────────────────────────────
            "imghash": report.get("imghash", {}),

            # ── Photoholmes (forgery detection library) ───────────
            "photoholmes": report.get("photoholmes", {}),

            # ── Similarity / hash comparison ──────────────────────
            "similar": report.get("similar", {}),

            # ── Crypto hashes ─────────────────────────────────────
            "hash": report.get("hash", {}),

            # ── Signatures ────────────────────────────────────────
            "signatures": report.get("signatures", []),
            "signature_count": len(report.get("signatures", [])),

            # ── File info ─────────────────────────────────────────
            "file_type": report.get("file_type"),
            "mime_type": report.get("mime_type"),
            "file_size": report.get("file_size"),

            # ── Raw EXIF / XMP (for deep review) ─────────────────
            "full_exif": exif_data,
            "full_xmp": xmp_data,
            "all_exif_keys": list(exif_data.keys()) if exif_data else [],
        }

        return data

    except Analysis.DoesNotExist:
        return {"error": f"Analysis {analysis_id} not found"}
    except Exception as e:
        return {"error": f"Error extracting analysis {analysis_id}: {str(e)}"}


def _extract_exif_summary(exif_data):
    """Pull key EXIF fields for quick review."""
    if not exif_data:
        return {}
    image_info = exif_data.get("Image", {})
    photo_info = exif_data.get("Photo", {})
    return {
        "make": image_info.get("Make"),
        "model": image_info.get("Model"),
        "software": image_info.get("Software"),
        "datetime": image_info.get("DateTime"),
        "orientation": image_info.get("Orientation"),
        "datetime_original": photo_info.get("DateTimeOriginal"),
        "datetime_digitized": photo_info.get("DateTimeDigitized"),
        "iso": photo_info.get("ISOSpeedRatings"),
        "exposure_time": photo_info.get("ExposureTime"),
        "f_number": photo_info.get("FNumber"),
        "focal_length": photo_info.get("FocalLength"),
        "flash": photo_info.get("Flash"),
    }


def _extract_software_info(exif_data, xmp_data):
    """Pull software / generator info — critical for AI detection."""
    info = {
        "exif_software": exif_data.get("Image", {}).get("Software") if exif_data else None,
        "exif_make": exif_data.get("Image", {}).get("Make") if exif_data else None,
        "exif_model": exif_data.get("Image", {}).get("Model") if exif_data else None,
    }
    if xmp_data:
        info["xmp_creator_tool"] = xmp_data.get("xmp", {}).get("CreatorTool")
        info["xmp_software"] = xmp_data.get("tiff", {}).get("Software")
    return info

def _print_summary(data):
    """Pretty-print a single-image summary to terminal."""
    if "error" in data:
        print(f"  ERROR: {data['error']}")
        return

    print(f"  File: {data['file_name']}")
    print(f"  State: {data['state']}")
    print(f"  Type: {data['file_type']}")
    print(f"  Size: {data.get('file_size', '?')} bytes")
    print(f"  Dimensions: {data['metadata']['dimensions']}")
    print(f"  Has EXIF: {data['metadata']['has_exif']}")
    print(f"  Has GPS: {data['metadata']['has_gps']}")

    # ── Auditor verdict (primary) ──
    aud = data.get("auditor", {})
    if aud.get("authenticity_score") is not None:
        print(f"\n  ── Auditor Verdict (Zero Trust) ──")
        print(f"    Authenticity Score: {aud['authenticity_score']}/100")
        print(f"    Verdict: {aud.get('verdict', 'N/A')}")
        print(f"    Label: {aud.get('verdict_label', 'N/A')}")
        print(f"    Confidence: {aud.get('verdict_confidence', 'N/A')}%")
        print(f"    Certainty: {aud.get('verdict_certainty', 'N/A')}")
        print(f"    Manipulation: {aud.get('manipulation_detected', 'N/A')}")
        if aud.get("detected_types"):
            print(f"    Detected Types: {', '.join(aud['detected_types'])}")

    # ── Audit metadata ──
    am = data.get("audit_metadata", {})
    if am:
        print(f"\n  ── Audit Metadata ──")
        print(f"    AI Probability: {am.get('ai_probability', 'N/A')}%")
        print(f"    Manipulation Probability: {am.get('manipulation_probability', 'N/A')}%")
        print(f"    Findings Count: {am.get('findings_count', 'N/A')}")

    # ── AI detection layers ──
    ai = data.get("ai_detection", {})
    if ai.get("enabled"):
        print(f"\n  ── AI Detection ──")
        print(f"    Framework: {ai.get('detection_framework', 'N/A')}")
        print(f"    Verdict: {ai.get('verdict', 'N/A')} ({ai.get('confidence', 'N/A')})")
        print(f"    Likely AI: {ai.get('likely_ai', 'N/A')}")
        print(f"    AI Probability: {ai.get('ai_probability', 'N/A')}%")
        layers = ai.get("detection_layers", [])
        if layers:
            print(f"    Detection Layers ({len(layers)}):")
            for layer in layers:
                v = layer.get("verdict", "")
                c = layer.get("confidence", "")
                s = layer.get("score", "")
                m = layer.get("method", "unknown")
                ev = layer.get("evidence", "")[:120]
                score_str = f" score={s:.4f}" if isinstance(s, (int, float)) else ""
                print(f"      [{m}] {v} ({c}{score_str}): {ev}")

    # ── Forensic methods ──
    freq = data.get("frequency_analysis", {})
    if freq:
        print(f"\n  ── Frequency Analysis ──")
        print(f"    Suspicious: {freq.get('suspicious', 'N/A')}")
        print(f"    Anomaly Score: {freq.get('anomaly_score', 'N/A')}")
        print(f"    Checkerboard: {freq.get('checkerboard_score', 'N/A')}")
        print(f"    Peak Ratio: {freq.get('peak_ratio', 'N/A')}")

    noise = data.get("noise_analysis", {})
    if noise:
        print(f"\n  ── Noise Analysis ──")
        print(f"    Suspicious: {noise.get('suspicious', 'N/A')}")
        print(f"    Inconsistency: {noise.get('inconsistency_score', 'N/A')}")
        print(f"    Anomaly Count: {noise.get('anomaly_count', 'N/A')}")

    ela = data.get("ela", {})
    if ela:
        print(f"\n  ── ELA ──")
        print(f"    Max Difference: {ela.get('max_difference', 'N/A')}")

    # ── OpenCV ──
    ocv = data.get("opencv_manipulation", {})
    if ocv.get("enabled"):
        print(f"\n  ── OpenCV (Computer Vision) ──")
        print(f"    Suspicious: {ocv.get('is_suspicious', 'N/A')}")
        print(f"    Overall Confidence: {ocv.get('overall_confidence', 'N/A')}")
        print(f"    Interpretation: {ocv.get('interpretation', 'N/A')}")
        md = ocv.get("manipulation_detection", {})
        if md:
            print(f"    Manipulation: {md.get('confidence', 'N/A'):.1%} ({md.get('num_anomalies', '?')} anomalies)")
        na = ocv.get("noise_analysis", {})
        if na:
            print(f"    Noise Consistency: {na.get('noise_consistency', 'N/A')}")
        ja = ocv.get("jpeg_artifacts", {})
        if ja:
            print(f"    JPEG Artifacts: {ja.get('confidence', 'N/A'):.1%}")

    # ── Perceptual hashes ──
    ih = data.get("imghash", {})
    if ih:
        print(f"\n  ── Perceptual Hashes ──")
        for k, v in ih.items():
            print(f"    {k}: {v}")

    # ── Photoholmes ──
    ph = data.get("photoholmes", {})
    if ph and ph.get("enabled"):
        print(f"\n  ── Photoholmes (Forgery Detection Library) ──")
        summary = ph.get("summary", {})
        if summary:
            print(f"    Methods Run: {summary.get('methods_run', 'N/A')}/{summary.get('methods_available', 'N/A')}")
            print(f"    Forgery Detected: {summary.get('forgery_detected_count', 0)} methods")
            print(f"    Avg Detection Score: {summary.get('avg_detection_score', 'N/A')}")
            print(f"    Max Detection Score: {summary.get('max_detection_score', 'N/A')}")
            print(f"    Consensus Forgery: {summary.get('consensus_forgery', 'N/A')}")
        
        methods = ph.get("methods", {})
        if methods:
            print(f"\n    Method Results:")
            for method_name, result in methods.items():
                if isinstance(result, dict) and not result.get("error"):
                    det_score = result.get("detection_score", "N/A")
                    forgery = result.get("forgery_detected", False)
                    forgery_str = "🚨 FORGERY" if forgery else "✓ Clean"
                    print(f"      [{method_name}] {forgery_str} - Score: {det_score}")
                    
                    # Show heatmap stats if available
                    hm_stats = result.get("heatmap_stats")
                    if hm_stats:
                        print(f"        Heatmap: mean={hm_stats.get('mean', 'N/A'):.4f}, "
                              f"max={hm_stats.get('max', 'N/A'):.4f}, "
                              f"suspicious_pixels={hm_stats.get('suspicious_pixel_ratio', 'N/A'):.1%}")
                    
                    # Show mask stats if available
                    mask_stats = result.get("mask_stats")
                    if mask_stats:
                        print(f"        Mask: forged_pixels={mask_stats.get('forged_pixel_ratio', 'N/A'):.1%}")
                elif isinstance(result, dict) and result.get("error"):
                    print(f"      [{method_name}] ERROR: {result['error']}")

    # ── Similarity ──
    sim = data.get("similar", {})
    if sim:
        print(f"\n  ── Similarity ──")
        print(f"    In-case matches: {len(sim.get('in_case', []))}")
        print(f"    Cross-case matches: {len(sim.get('cross_case', []))}")
        print(f"    Hash-list matches: {len(sim.get('hash_list_matches', []))}")
        if sim.get("inter_hash_distances"):
            print(f"    Inter-hash distances: {sim['inter_hash_distances']}")

    # ── Software signatures ──
    sw = data.get("software_info", {})
    if any(sw.values()):
        print(f"\n  ── Software Signatures ──")
        for k, v in sw.items():
            if v:
                print(f"    {k}: {v}")

    # ── Camera info ──
    exif = data.get("exif_summary", {})
    if exif.get("make") or exif.get("model"):
        print(f"\n  ── Camera ──")
        if exif.get("make"):
            print(f"    Make: {exif['make']}")
        if exif.get("model"):
            print(f"    Model: {exif['model']}")
        if exif.get("datetime"):
            print(f"    DateTime: {exif['datetime']}")

    # ── Confidence indicators (supporting, not authoritative) ──
    ci = data.get("confidence_indicators", {})
    indicators = ci.get("indicators", [])
    if indicators:
        print(f"\n  ── Confidence Indicators ({len(indicators)}) ──")
        for ind in indicators:
            print(f"    [{ind.get('type', '?')}] {ind.get('method', '?')}: {ind.get('evidence', '')[:120]}")

    # ── Signatures ──
    print(f"\n  ── Signatures ({data.get('signature_count', 0)}) ──")
    for sig in data.get("signatures", []):
        print(f"    [{sig.get('severity', '?')}] {sig.get('name', '?')}")


def main():
    """Extract data for specified images (or all if no args)."""

    # All known images post-reset
    all_images = {
        3: "airbrush-ivy.jpg (edited)",
        4: "psworkivy.jpg (photoshopped)",
        5: "mmmhmm.jpg",
        6: "cosplay.jpg",
        7: "cosplay2.jpg",
        8: "Gemini AI generated (PNG)",
        9: "PXL_20260109 - Pixel 8 Pro photo",
        10: "PXL_20260105 - Pixel 8 Pro photo",
    }

    # Parse CLI args — accept specific IDs or default to all
    if len(sys.argv) > 1:
        try:
            requested_ids = [int(a) for a in sys.argv[1:]]
        except ValueError:
            print(f"Usage: {sys.argv[0]} [analysis_id ...]")
            print(f"Available IDs: {', '.join(str(k) for k in all_images)}")
            sys.exit(1)
        test_images = {k: all_images.get(k, f"Image {k}") for k in requested_ids}
    else:
        test_images = all_images

    output_dir = os.path.dirname(os.path.abspath(__file__))

    for img_id, description in test_images.items():
        print(f"\n{'=' * 70}")
        print(f"Analysis #{img_id}: {description}")
        print(f"{'=' * 70}")

        data = extract_analysis_data(img_id)

        # Save to JSON
        output_file = os.path.join(output_dir, f"analysis_{img_id}.json")
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"✓ Saved to: {output_file}")
        _print_summary(data)

    print(f"\n{'=' * 70}")
    print(f"Data extracted to: {output_dir}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
