"""Jev vs Gemini on the 50 human-labelled feedback entries.

    cd backend && python ../eval/benchmark.py            # all available models
    python ../eval/benchmark.py --repeats 3               # also measures consistency

Metrics: category accuracy, severity MAE, churn precision/recall (P(yes) >= 0.5),
median latency, consistency across repeats. Cost is estimated from token
counts you fill in from each provider's pricing page (see PRICES).

Do not assume Jev wins. Whatever the numbers say goes in the README.
"""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app import config  # noqa: E402
from app.services import jev_service, llm_service  # noqa: E402

DATA = ROOT / "data" / "sample_feedback.csv"
OUT = ROOT / "eval" / "results.md"


def run_jev(text):
    d = jev_service.decide(text)
    return {"category": d["category"], "severity": d["severity"],
            "churn": d["churn_probability"], "latency_ms": d["latency_ms"], "model": d["model_used"]}


def run_baseline(text):
    d = jev_service.mock_decide(text)
    return {"category": d["category"], "severity": d["severity"],
            "churn": d["churn_probability"], "latency_ms": 0.0, "model": "keyword-baseline"}


def run_gemini(text):
    d = llm_service.gemini_classify(text)
    return {"category": d.get("category"), "severity": float(d.get("severity", 1)),
            "churn": float(d.get("churn_risk", 0)), "latency_ms": d["latency_ms"], "model": config.GEMINI_MODEL}


def score(rows, preds):
    acc = sum(p["category"] == r["label_category"] for r, p in zip(rows, preds)) / len(rows)
    mae = statistics.mean(abs(p["severity"] - float(r["label_severity"])) for r, p in zip(rows, preds))
    tp = sum(p["churn"] >= 0.5 and r["label_churn"] == "1" for r, p in zip(rows, preds))
    fp = sum(p["churn"] >= 0.5 and r["label_churn"] == "0" for r, p in zip(rows, preds))
    fn = sum(p["churn"] < 0.5 and r["label_churn"] == "1" for r, p in zip(rows, preds))
    return {
        "category_accuracy": round(acc, 3),
        "severity_mae": round(mae, 2),
        "churn_precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "churn_recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "median_latency_ms": round(statistics.median(p["latency_ms"] for p in preds), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(DATA)))
    # The keyword baseline is always reported: any model has to beat it to justify its cost.
    runners = {"keyword-baseline": run_baseline}
    if config.jev_live():
        runners["jev"] = run_jev
    if config.gemini_live():
        runners["gemini"] = run_gemini

    report = {}
    for name, fn in runners.items():
        runs = []
        for _ in range(args.repeats):
            preds = []
            for r in rows:
                try:
                    preds.append(fn(r["content"]))
                except Exception as e:  # count failures as wrong, don't crash the benchmark
                    preds.append({"category": None, "severity": 3, "churn": 0, "latency_ms": 0, "error": str(e)})
                time.sleep(0.05)
            runs.append(preds)
        m = score(rows, runs[0])
        if args.repeats > 1:
            same = sum(all(run[i]["category"] == runs[0][i]["category"] for run in runs) for i in range(len(rows)))
            m["category_consistency"] = round(same / len(rows), 3)
        m["errors"] = sum("error" in p for p in runs[0])
        report[name] = m
        # Per-row disagreements, useful for the README "where it failed" section
        m["misclassified"] = [{"text": r["content"][:80], "label": r["label_category"], "pred": p["category"]}
                              for r, p in zip(rows, runs[0]) if p["category"] != r["label_category"]][:10]

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "misclassified"} for k, v in report.items()}, indent=2))
    lines = ["# Benchmark results", "", f"Dataset: {len(rows)} human-labelled entries (`data/sample_feedback.csv`)", "",
             "| Model | Category acc. | Severity MAE | Churn precision | Churn recall | Median latency (ms) | Consistency |",
             "|---|---|---|---|---|---|---|"]
    for k, v in report.items():
        lines.append(f"| {k} | {v['category_accuracy']} | {v['severity_mae']} | {v['churn_precision']} | "
                     f"{v['churn_recall']} | {v['median_latency_ms']} | {v.get('category_consistency', '—')} |")
    for k, v in report.items():
        lines += ["", f"## Misclassified by {k} (first 10)", ""] + \
                 [f"- \"{x['text']}\" — label `{x['label']}`, predicted `{x['pred']}`" for x in v["misclassified"]]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
