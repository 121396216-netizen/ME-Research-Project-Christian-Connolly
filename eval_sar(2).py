"""
SAR Robot Navigation Classifier — Evaluation Script
=====================================================
Tests a local Ollama model (default: gemma3:4b) against 40 labelled
scene descriptions and reports per-category + overall accuracy.

Usage:
    pip install requests
    # Make sure Ollama is running with your model pulled:
    #   ollama pull gemma3:4b
    python eval_sar.py                        # defaults
    python eval_sar.py --model gemma3:4b      # explicit model
    python eval_sar.py --runs 3               # average over N runs
"""

import json
import time
import argparse
import requests
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """You are a SAR (search and rescue) robot navigation classifier. Output only one letter — nothing else."""

USER_PROMPT_TEMPLATE = """Classify this scene for a SAR robot. Output exactly one letter.
P = a person, human, man, woman, child, or body is visible
S = hazard present (fire, smoke, flood, collapse, sparks, power line, gas leak, caution tape) — no person
R = no person explicitly visible but signs of human presence are present (e.g. clothing, bags, shoes, phone, food, drink, accessories, blood, stains) — no hazard
F = none of the above apply — clear scene with no people, hazards, or personal items

Scene: {scene}
Letter:"""

VALID_LABELS = {"P", "S", "R", "F"}

# ── Helpers ─────────────────────────────────────────────────────────────────

def call_ollama(model: str, scene: str, timeout: int = 120) -> dict:
    """Send a single scene to the Ollama API and return raw + parsed label."""
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": USER_PROMPT_TEMPLATE.format(scene=scene),
        "stream": False,
        "options": {
            "temperature": 0.0,   # deterministic
            "num_predict": 4,     # we only need 1 token
        },
    }
    t0 = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    raw = data.get("response", "").strip()

    # Parse: take the first valid letter found
    parsed = None
    for ch in raw.upper():
        if ch in VALID_LABELS:
            parsed = ch
            break

    return {"raw": raw, "parsed": parsed, "time_s": round(elapsed, 2)}


def run_eval(test_cases: list, model: str) -> list:
    """Run all test cases and return results list."""
    results = []
    for tc in test_cases:
        tid = tc["id"]
        expected = tc["expected"].upper()
        scene = tc["scene"]

        difficulty = tc.get("difficulty", "unknown")

        try:
            out = call_ollama(model, scene)
            predicted = out["parsed"]
            correct = predicted == expected
            results.append({
                "id": tid,
                "expected": expected,
                "predicted": predicted,
                "difficulty": difficulty,
                "raw": out["raw"],
                "correct": correct,
                "time_s": out["time_s"],
            })
            status = "✓" if correct else "✗"
            print(f"  [{status}] #{tid:>2}  expected={expected}  got={predicted or '???'}  "
                  f"(raw=\"{out['raw']}\")  {out['time_s']}s")
        except Exception as e:
            results.append({
                "id": tid,
                "expected": expected,
                "predicted": None,
                "difficulty": difficulty,
                "raw": str(e),
                "correct": False,
                "time_s": 0,
            })
            print(f"  [!] #{tid:>2}  ERROR: {e}")

    return results


def print_report(results: list, run_label: str = ""):
    """Print accuracy breakdown."""
    header = f"\n{'='*60}\n  RESULTS {run_label}\n{'='*60}"
    print(header)

    by_cat = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        cat = r["expected"]
        by_cat[cat]["total"] += 1
        if r["correct"]:
            by_cat[cat]["correct"] += 1

    total_correct = sum(r["correct"] for r in results)
    total = len(results)

    print(f"\n  {'Category':<12} {'Correct':<10} {'Total':<8} {'Accuracy'}")
    print(f"  {'-'*44}")
    for cat in sorted(by_cat):
        c = by_cat[cat]["correct"]
        t = by_cat[cat]["total"]
        pct = (c / t * 100) if t else 0
        print(f"  {cat:<12} {c:<10} {t:<8} {pct:.1f}%")

    overall = (total_correct / total * 100) if total else 0
    print(f"  {'-'*44}")
    print(f"  {'OVERALL':<12} {total_correct:<10} {total:<8} {overall:.1f}%")

    avg_time = sum(r["time_s"] for r in results) / total if total else 0
    print(f"\n  Avg response time: {avg_time:.2f}s")

    # Accuracy by difficulty
    by_diff = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        d = r.get("difficulty", "unknown")
        by_diff[d]["total"] += 1
        if r["correct"]:
            by_diff[d]["correct"] += 1

    if any(d != "unknown" for d in by_diff):
        print(f"\n  {'Difficulty':<12} {'Correct':<10} {'Total':<8} {'Accuracy'}")
        print(f"  {'-'*44}")
        for diff in ["easy", "medium", "hard"]:
            if diff in by_diff:
                c = by_diff[diff]["correct"]
                t = by_diff[diff]["total"]
                pct = (c / t * 100) if t else 0
                print(f"  {diff:<12} {c:<10} {t:<8} {pct:.1f}%")

    # Confusion details
    misses = [r for r in results if not r["correct"]]
    if misses:
        print(f"\n  Misclassifications ({len(misses)}):")
        for r in misses:
            print(f"    #{r['id']:>2}  expected={r['expected']}  got={r['predicted'] or 'PARSE_FAIL'}  "
                  f"raw=\"{r['raw']}\"")
    else:
        print("\n  Perfect score — no misclassifications!")

    return overall


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate SAR classifier model")
    parser.add_argument("--model", default="gemma3:4b-it-q4_K_M",
                        help="Ollama model name (default: gemma3:4b-it-q4_K_M for 4-bit quant)")
    parser.add_argument("--test-file", default="sar_test_cases.json",
                        help="Path to test cases JSON (default: sar_test_cases.json)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of evaluation runs to average (default: 1)")
    parser.add_argument("--output", default="sar_results.json",
                        help="Path to save detailed results JSON (default: sar_results.json)")
    args = parser.parse_args()

    with open(args.test_file, "r") as f:
        test_cases = json.load(f)

    print(f"\nSAR Classifier Evaluation")
    print(f"Model   : {args.model}")
    print(f"Quant   : 4-bit (Q4_K_M)" if "q4" in args.model.lower() else "")
    print(f"Cases   : {len(test_cases)}")
    print(f"Runs    : {args.runs}\n")

    all_runs = []
    accuracies = []

    for run_idx in range(1, args.runs + 1):
        label = f"(Run {run_idx}/{args.runs})" if args.runs > 1 else ""
        print(f"\n--- Run {run_idx} {'-'*45}")
        results = run_eval(test_cases, args.model)
        acc = print_report(results, label)
        accuracies.append(acc)
        all_runs.append({"run": run_idx, "results": results, "accuracy": acc})

    if args.runs > 1:
        avg_acc = sum(accuracies) / len(accuracies)
        min_acc = min(accuracies)
        max_acc = max(accuracies)
        print(f"\n{'='*60}")
        print(f"  SUMMARY OVER {args.runs} RUNS")
        print(f"  Mean accuracy : {avg_acc:.1f}%")
        print(f"  Min  accuracy : {min_acc:.1f}%")
        print(f"  Max  accuracy : {max_acc:.1f}%")
        print(f"{'='*60}")

    # Save detailed results
    with open(args.output, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
