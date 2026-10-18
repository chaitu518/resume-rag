"""Evaluation: retrieval accuracy, metadata accuracy and latency.

Relevance labels come from data/ground_truth.json (graded):
  2 = right role family AND meets the JD's minimum years  (ideal hire)
  1 = right role family but too junior                     (partially relevant)
  0 = everything else

Metrics (per JD, then averaged):
  R-Prec     precision at R, where R = number of grade-2 candidates for the JD
  R@10       share of all grade-2 candidates found in top-10
  MRR        1 / rank of first grade-2 candidate
  nDCG@10    graded ranking quality
  Family@10  share of top-10 from the right role family (grade >= 1), unfiltered runs
"""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from .job_matcher import JobMatcher
from .resume_rag import ROOT

GT_PATH = ROOT / "data" / "ground_truth.json"


def load_ground_truth() -> dict:
    return json.loads(GT_PATH.read_text())


def _dcg(gains):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ranking_metrics(ranked_paths: list[str], relevance: dict[str, int], k: int = 10) -> dict:
    top = ranked_paths[:k]
    gains = [relevance.get(p, 0) for p in top]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    n_ideal = sum(1 for v in relevance.values() if v == 2)
    first = next((i + 1 for i, g in enumerate(gains) if g == 2), None)
    return {
        "R-Prec": (sum(1 for g in gains[:n_ideal] if g == 2) / n_ideal) if n_ideal else 0.0,
        "R@10": (sum(1 for g in gains if g == 2) / n_ideal) if n_ideal else 0.0,
        "MRR": 1 / first if first else 0.0,
        "nDCG@10": _dcg(gains) / _dcg(ideal) if ideal else 0.0,
        "Family@10": sum(1 for g in gains if g >= 1) / max(1, min(k, len(top))),
    }


def evaluate(matcher: JobMatcher, mode: str = "hybrid", apply_filters: bool = True, k: int = 10) -> dict:
    gt = load_ground_truth()
    per_jd, latencies = {}, []
    for jd_id, info in gt["job_descriptions"].items():
        text = (ROOT / info["path"]).read_text()
        t0 = time.perf_counter()
        res = matcher.match(text, k=k, mode=mode, apply_filters=apply_filters)
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = [m["resume_path"] for m in res["top_matches"]]
        m = ranking_metrics(ranked, info["relevance"], k)
        # must-have filter precision: of returned candidates, how many truly meet years + family
        truth = gt["candidates"]
        ok = [truth[p]["years"] >= info["min_years"] for p in ranked]
        m["meets_years@k"] = sum(ok) / len(ok) if ok else 0.0
        per_jd[jd_id] = m
    keys = next(iter(per_jd.values())).keys()
    avg = {k_: round(statistics.mean(r[k_] for r in per_jd.values()), 3) for k_ in keys}
    return {"mode": mode, "filters": apply_filters, "embedder": matcher.rag.embedder.name,
            "mean": avg, "per_jd": per_jd, "latency_ms": latency_summary(latencies)}


def evaluate_paraphrased(matcher: JobMatcher, mode: str = "hybrid", k: int = 5) -> dict:
    """Short recruiter-style queries with few literal skill names; relevant = right family."""
    gt = load_ground_truth()["paraphrased_queries"]
    per = {}
    for qid, q in gt.items():
        res = matcher.match(q["text"], k=k, mode=mode, apply_filters=False)
        ranked = [m["resume_path"] for m in res["top_matches"]]
        rel = q["relevance"]
        first = next((i + 1 for i, p in enumerate(ranked) if p in rel), None)
        per[qid] = {"P@3": sum(p in rel for p in ranked[:3]) / 3,
                    f"P@{k}": sum(p in rel for p in ranked) / k, "MRR": 1 / first if first else 0.0}
    keys = next(iter(per.values())).keys()
    return {"mode": mode, "mean": {k_: round(statistics.mean(r[k_] for r in per.values()), 3) for k_ in keys},
            "per_query": per}


def latency_summary(samples: list[float]) -> dict:
    s = sorted(samples)
    pct = lambda q: s[min(len(s) - 1, int(round(q * (len(s) - 1))))]
    return {"n": len(s), "mean": round(statistics.mean(s), 1), "p50": round(pct(0.5), 1),
            "p95": round(pct(0.95), 1), "max": round(s[-1], 1)}


def benchmark_latency(matcher: JobMatcher, repeats: int = 20, mode: str = "hybrid") -> dict:
    gt = load_ground_truth()
    texts = [(ROOT / i["path"]).read_text() for i in gt["job_descriptions"].values()]
    matcher.match(texts[0])  # warm-up
    samples = []
    for _ in range(repeats):
        for t in texts:
            t0 = time.perf_counter()
            matcher.match(t, mode=mode)
            samples.append((time.perf_counter() - t0) * 1000)
    return latency_summary(samples)


def metadata_accuracy(registry: dict) -> dict:
    """Compare extracted metadata with the generator's ground truth."""
    truth = load_ground_truth()["candidates"]
    name_ok, years_err, skill_recall, degree_ok = 0, [], [], 0
    rows = []
    for path, t in truth.items():
        m = registry.get(path)
        if not m:
            continue
        name_ok += m["candidate_name"] == t["name"]
        years_err.append(abs(m["experience_years"] - t["years"]))
        got = set(m["skills"])
        skill_recall.append(len(got & set(t["skills"])) / len(t["skills"]))
        deg = t["degree"].lower()
        expected = ("PhD" if "ph.d" in deg else "Master's" if deg.startswith(("m.", "mba", "mca"))
                    else "Diploma" if deg.startswith("diploma") else "Bachelor's")
        degree_ok += m["highest_degree"] == expected
        rows.append({"resume": Path(path).name, "years_true": t["years"], "years_extracted": m["experience_years"],
                     "skill_recall": round(skill_recall[-1], 2), "degree": m["highest_degree"]})
    n = len(rows)
    return {"n": n, "name_accuracy": round(name_ok / n, 3),
            "years_MAE": round(statistics.mean(years_err), 2),
            "years_within_0.5": round(sum(e <= 0.5 for e in years_err) / n, 3),
            "skill_recall": round(statistics.mean(skill_recall), 3),
            "degree_accuracy": round(degree_ok / n, 3), "rows": rows}
