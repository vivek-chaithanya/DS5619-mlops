"""
Driver script — wires drift_monitor.py into the flow this lab is about.
Complete, don't edit. Run with:

    python src/run_pipeline.py

Treats camera_A_daylight as the "reference" distribution (what the detector
was implicitly tuned against) and camera_B_lowlight as the "live" traffic
you're monitoring — an illustrative stand-in for the same train/deploy
mismatch behind BMD-45's real domain-shift finding. Writes
drift_report.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drift_monitor as dm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_CAMERA = os.path.join(REPO_ROOT, "data", "fixtures", "camera_A_daylight")
LIVE_CAMERA = os.path.join(REPO_ROOT, "data", "fixtures", "camera_B_lowlight")


def main():
    reference_scores = dm.extract_confidence_scores(REFERENCE_CAMERA)
    live_scores = dm.extract_confidence_scores(LIVE_CAMERA)

    if not reference_scores or not live_scores:
        raise SystemExit(
            "No detections found in one of the camera directories — "
            "check extract_confidence_scores before continuing."
        )

    psi = dm.compute_psi(reference_scores, live_scores)
    drift_level = dm.classify_drift(psi)

    report = {
        "reference_camera": "camera_A_daylight",
        "live_camera": "camera_B_lowlight",
        "reference_summary": dm.summarize_scores(reference_scores),
        "live_summary": dm.summarize_scores(live_scores),
        "psi": round(psi, 4),
        "drift_level": drift_level,
        "thresholds": {
            "moderate": dm.PSI_MODERATE_THRESHOLD,
            "significant": dm.PSI_SIGNIFICANT_THRESHOLD,
        },
    }

    out_path = os.path.join(REPO_ROOT, "drift_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Reference ({report['reference_camera']}): {report['reference_summary']}")
    print(f"Live ({report['live_camera']}): {report['live_summary']}")
    print(f"PSI = {report['psi']} -> drift level: {drift_level}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
