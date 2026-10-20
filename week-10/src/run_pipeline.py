"""
Driver script — wires fairness_audit.py into the flow this lab is about.
Complete, don't edit. Run with:

    python src/run_pipeline.py

Runs the class-based performance disparity audit across all 14 BMD-45
vehicle categories (synthetic fixtures), then re-runs it under an
FGSM-style colour perturbation at several budgets, and writes
fairness_report.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fairness_audit as fa
import adversarial as adv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(REPO_ROOT, "data", "fixtures")
COCO_PATH = os.path.join(FIXTURES_DIR, "_annotations.coco.json")
RECALL_THRESHOLD = 0.90
ATTACK_EPS = (0, 10, 20, 30, 40)   # L-infinity budgets, 0-255 scale
REPORT_EPS = 20                     # budget used for the "most fragile" ranking


def main():
    category_stats = fa.audit_by_category(COCO_PATH, FIXTURES_DIR)
    underperforming = fa.find_underperforming_categories(category_stats, RECALL_THRESHOLD)

    attack_stats = adv.audit_under_attack(COCO_PATH, FIXTURES_DIR, ATTACK_EPS)
    fragile = adv.find_most_fragile(attack_stats, REPORT_EPS, top_k=3)
    # Absolute view: which classes have the LOWEST recall once attacked?
    after = attack_stats[REPORT_EPS]
    lowest = sorted(after, key=lambda n: after[n]["recall"])[:3]

    report = {
        "recall_threshold": RECALL_THRESHOLD,
        "category_stats": category_stats,
        "underperforming_categories": [
            {"category": name, "recall": recall} for name, recall in underperforming
        ],
        "adversarial": {
            "eps_values": list(ATTACK_EPS),
            "recall_by_eps": {
                str(eps): {name: s["recall"] for name, s in stats.items()}
                for eps, stats in attack_stats.items()
            },
            "report_eps": REPORT_EPS,
            "most_fragile": [{"category": n, "recall_drop": d} for n, d in fragile],
            "lowest_recall_under_attack": lowest,
        },
    }

    out_path = os.path.join(REPO_ROOT, "fairness_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Per-category recall:")
    for name, stats in sorted(category_stats.items(), key=lambda kv: kv[1]["recall"]):
        flag = " <-- below threshold" if stats["recall"] < RECALL_THRESHOLD else ""
        print(f"  {name:18s} {stats['detected']:3d}/{stats['total']:3d} = {stats['recall']:.1%}{flag}")
    print(f"\n{len(underperforming)} categories below {RECALL_THRESHOLD:.0%} recall threshold.")

    print(f"\nRecall under attack (rows = classes, columns = eps {list(ATTACK_EPS)}):")
    for name in sorted(category_stats, key=lambda n: category_stats[n]["recall"]):
        row = "  ".join(f"{attack_stats[e][name]['recall']:.2f}" for e in ATTACK_EPS)
        print(f"  {name:18s} {row}")
    print(f"\nMost fragile at eps={REPORT_EPS}: " + ", ".join(f"{n} (-{d:.0%})" for n, d in fragile))
    print("Lowest recall at that budget: " + ", ".join(f"{n} ({after[n]['recall']:.0%})" for n in lowest))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main() 
