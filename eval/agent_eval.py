"""Hidden-cause evaluation for the Investigator agent.

    cd backend && python ../eval/agent_eval.py            # agent (needs Gemini key) + scripted baseline
    python ../eval/agent_eval.py --runs 3                 # repeat each case to measure stability

For each case: wipe the DB, ingest planted + noise reports through the normal
pipeline, pick the open issue holding the most planted reports, then run
(a) the agent and (b) the scripted baseline on it. Score with the case's
`must_mention` rule. Writes eval/agent_results.md.

Note: scoring is keyword-based and transparent, so it can be strict (a correct
cause phrased oddly fails). Read the briefs in the output, not only the score.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app import config  # noqa: E402
from app.db import Base, Feedback, ProductIssue, SessionLocal, engine  # noqa: E402
from app.services import agent_service, pipeline  # noqa: E402

CASES = json.loads((ROOT / "eval" / "agent_cases.json").read_text())["cases"]
OUT = ROOT / "eval" / "agent_results.md"


def score(report, case) -> bool:
    if not report:
        return False
    text = f"{report.get('root_cause', '')} {report.get('affected_segment', '')}".lower()
    return all(any(t in text for t in group) for group in case["must_mention"])


def setup(case):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    s = SessionLocal()
    planted_ids = set()
    rows = [(t, True) for t in case["reports"]] + [(t, False) for t in case["noise"]]
    random.Random(7).shuffle(rows)
    for text, planted in rows:
        r = pipeline.process(s, text, "eval")
        if planted:
            planted_ids.add(r["feedback_id"])
    # target = the open issue that absorbed the most planted reports
    counts = {}
    for f in s.query(Feedback).filter(Feedback.id.in_(planted_ids)):
        if f.issue_id:
            counts[f.issue_id] = counts.get(f.issue_id, 0) + 1
    target = s.get(ProductIssue, max(counts, key=counts.get))
    purity = counts[target.id] / target.feedback_count
    return s, target, counts[target.id], len(planted_ids), purity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()
    modes = ["scripted"] + (["agent"] if config.gemini_live() else [])
    if "agent" not in modes:
        print("No GEMINI_API_KEY: running scripted baseline only.\n")
    rows, briefs = [], []
    for case in CASES:
        for mode in modes:
            for run_n in range(args.runs if mode == "agent" else 1):
                s, issue, caught, planted, purity = setup(case)
                if mode == "agent":
                    run = agent_service.investigate(s, issue.id)
                else:
                    run = agent_service.investigate(s, issue.id, llm=None)
                v = run.verification or {}
                tools = sum(1 for st in (run.steps or []) if st.get("tool"))
                hit = score(run.report, case)
                rows.append({"case": case["name"], "mode": run.mode.split(" ")[0], "run": run_n + 1,
                             "found_cause": hit, "status": run.status, "tool_calls": tools,
                             "verifier_rejections": v.get("verifier_rejections", 0),
                             "fabricated_ids_caught": len(v.get("fabricated_ids_caught", [])),
                             "support": v.get("support_probability"), "seconds": round((run.duration_ms or 0) / 1000, 1),
                             "issue_purity": round(purity, 2), "planted_in_issue": f"{caught}/{planted}"})
                briefs.append((case, mode, run))
                print(json.dumps(rows[-1]))
                s.close()

    def rate(mode):
        r = [x for x in rows if x["mode"] == mode]
        return f"{sum(x['found_cause'] for x in r)}/{len(r)}" if r else "—"

    lines = ["# Investigator agent: hidden-cause evaluation", "",
             f"{len(CASES)} scenarios with a planted root cause hidden among same-area noise (`eval/agent_cases.json`).", "",
             f"**Root cause found — agent: {rate('agent')} · scripted baseline: {rate('scripted')}**", "",
             "| Case | Mode | Run | Found cause | Status | Tool calls | Verifier rejections | Fabricated IDs caught | Final support | Seconds | Planted reports in issue |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        lines.append(f"| {x['case']} | {x['mode']} | {x['run']} | {'✅' if x['found_cause'] else '❌'} | {x['status']} | "
                     f"{x['tool_calls']} | {x['verifier_rejections']} | {x['fabricated_ids_caught']} | {x['support']} | "
                     f"{x['seconds']} | {x['planted_in_issue']} |")
    lines += ["", "## Briefs", ""]
    for case, mode, run in briefs:
        rep = run.report or {}
        lines += [f"### {case['name']} — {mode}", f"- Planted: {case['planted_cause']}",
                  f"- Found: {rep.get('root_cause', '(no report)')} — segment: {rep.get('affected_segment', '—')}",
                  f"- Path: {' → '.join(st['tool'] for st in (run.steps or []) if st.get('tool'))}", ""]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nagent: {rate('agent')}   scripted: {rate('scripted')}\nWrote {OUT}")


if __name__ == "__main__":
    main()
