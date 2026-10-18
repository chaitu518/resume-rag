"""Part B - Job matching engine.

match(jd_text) does:
  1. Parse the JD: skills, must-have skills, minimum years, nice-to-haves.
  2. Must-have filter: pushed down into ChromaDB as a `where` clause on
     per-skill flags and experience_years, so ineligible resumes are never
     retrieved. Excluded candidates are still reported with the reason.
  3. Hybrid retrieval: semantic (JD embedding vs chunk embeddings) UNION
     keyword (BM25 over chunks, query = critical skills + aliases).
  4. Score 0-100 per candidate:
        40% semantic similarity (calibrated per embedding model)
        15% BM25 keyword strength
        35% weighted skill coverage (must-have x2, required x1, nice x0.5)
        10% experience fit vs required years
  5. Explain: matched skills, missing skills, matched sections, excerpts.

CLI:
    python -m api.job_matcher --jd data/job_descriptions/jd01_senior_ml_engineer.txt
    python -m api.job_matcher --jd "We need a Python dev with 3+ years Django" --k 5 --mode semantic
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .llm import generate_reasoning, llm_available
from .resume_rag import ResumeRAG
from .skills_taxonomy import extract_skills, skill_key, skill_patterns

WEIGHTS = {"semantic": 0.40, "keyword": 0.15, "skills": 0.35, "experience": 0.10}
_YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:years|yrs)", re.I)
_TOKEN_RE = re.compile(r"[a-z0-9+#][a-z0-9+#./-]*")


def tokenize(text: str) -> list[str]:
    return [t.strip("./-") for t in _TOKEN_RE.findall(text.lower()) if t.strip("./-")]


# ---------------------------------------------------------------- JD parsing
def parse_job_description(text: str) -> dict:
    lines = [l.strip() for l in text.split("\n")]
    must: dict[str, int | None] = {}
    nice: set[str] = set()
    min_years = 0
    in_must_block = False
    for line in lines:
        low = line.lower()
        if not line:
            in_must_block = False
            continue
        if re.match(r"^(must[- ]haves?|required|mandatory)\s*:?\s*$", low):
            in_must_block = True
            continue
        if line.endswith(":") and len(line) < 40:
            in_must_block = False
        # "Nice to have: A, B, C" / "Must have: A, B" label the whole line
        if re.match(r"^(nice[- ]to[- ]have|bonus|preferred|good to have)\b", low):
            nice.update(s for s in extract_skills(line) if s not in must)
            continue
        if re.match(r"^(must[- ]haves?|required|mandatory)\s*:", low):
            for s in extract_skills(line):
                must.setdefault(s, None)
        # analyse clause by clause so "4+ years, Kubernetes required, AWS a plus" on one line parses correctly
        for clause in re.split(r"[;,.](?=\s)|\s(?:and|but)\s(?=\w+\s+(?:a plus|nice|preferred))", line):
            cl = clause.lower()
            cl_skills = extract_skills(clause)
            yrs = _YEARS_RE.search(clause)
            if yrs:
                y = int(yrs.group(1))
                min_years = max(min_years, y)
                for s in cl_skills:
                    must[s] = y
            if in_must_block or re.search(r"must[- ]have|is required|are required|mandatory", cl):
                for s in cl_skills:
                    must.setdefault(s, None)
            if re.search(r"nice to have|a plus|bonus|preferred", cl):
                nice.update(s for s in cl_skills if s not in must)
    skills = extract_skills(text)
    required = [s for s in skills if s not in must and s not in nice]
    return {"skills": skills, "must_have": must, "required": required, "nice_to_have": sorted(nice),
            "min_years": min_years}


# ---------------------------------------------------------------- matcher
class JobMatcher:
    def __init__(self, rag: ResumeRAG | None = None):
        self.rag = rag or ResumeRAG()
        self.registry = self.rag.load_registry()
        data = self.rag.all_chunks()
        self.chunk_ids = data["ids"]
        self.chunk_docs = data["documents"]
        self.chunk_meta = data["metadatas"]
        self.bm25 = BM25Okapi([tokenize(d) for d in self.chunk_docs]) if self.chunk_docs else None

    # -- must-have handling
    def _where_clause(self, jd: dict) -> dict | None:
        conds = [{skill_key(s): True} for s in jd["must_have"]]
        if jd["min_years"]:
            conds.append({"experience_years": {"$gte": float(jd["min_years"])}})
        if not conds:
            return None
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def _must_have_check(self, meta: dict, jd: dict) -> list[str]:
        reasons = []
        for s, yrs in jd["must_have"].items():
            if s not in meta["skills"]:
                reasons.append(f"missing must-have skill: {s}")
        if jd["min_years"] and meta["experience_years"] < jd["min_years"]:
            reasons.append(f"{meta['experience_years']} yrs experience < required {jd['min_years']}+")
        return reasons

    # -- retrieval signals
    def _semantic(self, jd_text: str, where: dict | None, n_chunks: int):
        q = self.rag.embedder.embed_query(jd_text)
        res = self.rag.query(q, n_results=n_chunks, where=where)
        by_cand = defaultdict(list)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            by_cand[meta["resume_path"]].append({"text": doc, "section": meta["section"], "sim": 1 - dist})
        return by_cand

    def _keyword(self, jd: dict, eligible: set[str] | None):
        if self.bm25 is None:
            return {}
        query = []
        for s in jd["skills"]:
            toks = []
            for alias in skill_patterns(s):
                toks.extend(tokenize(alias))
            reps = 2 if s in jd["must_have"] else (1 if s not in jd["nice_to_have"] else 1)
            query.extend(toks * reps)
        scores = self.bm25.get_scores(query) if query else np.zeros(len(self.chunk_docs))
        per = defaultdict(float)
        for sc, meta in zip(scores, self.chunk_meta):
            p = meta["resume_path"]
            if eligible is None or p in eligible:
                per[p] += float(sc)
        return per

    def _skill_coverage(self, meta: dict, jd: dict):
        cand = set(meta["skills"])
        weights = {s: 2.0 for s in jd["must_have"]}
        weights.update({s: 1.0 for s in jd["required"]})
        weights.update({s: 0.5 for s in jd["nice_to_have"]})
        total = sum(weights.values()) or 1.0
        got = sum(w for s, w in weights.items() if s in cand)
        matched = [s for s in jd["skills"] if s in cand]
        missing = [s for s in jd["skills"] if s not in cand and s not in jd["nice_to_have"]]
        return got / total, matched, missing

    # -- explanation helpers
    @staticmethod
    def _excerpts(chunks: list[dict], skills: list[str], limit: int = 3) -> list[str]:
        pats = [re.compile(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", re.I)
                for s in skills for a in skill_patterns(s)]
        out = []
        # prefer evidence of doing the work (experience/projects) over the bare skills list
        rank = {"experience": 0, "projects": 1, "summary": 2, "certifications": 3, "skills": 4}
        for ch in sorted(chunks, key=lambda c: (rank.get(c["section"], 5), -c["sim"])):
            body = ch["text"].split("\n", 1)[1] if " | " in ch["text"].split("\n", 1)[0] else ch["text"]
            for line in body.split("\n"):
                line = line.strip().lstrip("-• ").strip()
                if len(line) > 40 and any(p.search(line) for p in pats) and line not in out:
                    out.append(line[:220])
                if len(out) >= limit:
                    return out
        if not out and chunks:
            out.append(chunks[0]["text"].split("\n")[-1][:220])
        return out

    @staticmethod
    def _template_reasoning(score, meta, jd, matched, missing, sections) -> str:
        tier = ("Strong match" if score >= 80 else "Good match" if score >= 65
                else "Partial match" if score >= 45 else "Weak match")
        parts = [f"{tier}: {meta['experience_years']} yrs experience"
                 + (f" (role asks for {jd['min_years']}+)" if jd["min_years"] else "")]
        if matched:
            parts.append(f"covers {len(matched)}/{len(jd['skills'])} JD skills ({', '.join(matched[:6])})")
        if missing:
            parts.append(f"gaps: {', '.join(missing[:4])}")
        if sections:
            parts.append(f"strongest evidence in {', '.join(sections)}")
        return "; ".join(parts) + "."

    # -- main entry point
    def match(self, jd_text: str, k: int = 10, mode: str = "hybrid", apply_filters: bool = True,
              use_llm: bool = False, n_chunks: int = 200) -> dict:
        t0 = time.perf_counter()
        jd = parse_job_description(jd_text)

        excluded = []
        eligible = set(self.registry)
        if apply_filters and (jd["must_have"] or jd["min_years"]):
            for path, meta in self.registry.items():
                r = self._must_have_check(meta, jd)
                if r:
                    eligible.discard(path)
                    excluded.append({"candidate_name": meta["candidate_name"], "resume_path": path, "reasons": r})
        where = self._where_clause(jd) if apply_filters else None

        sem = self._semantic(jd_text, where, n_chunks) if eligible else {}
        t_sem = time.perf_counter()
        kw = self._keyword(jd, eligible)
        t_kw = time.perf_counter()

        floor, ceil = self.rag.embedder.sim_floor, self.rag.embedder.sim_ceil
        kw_max = max(kw.values()) if kw else 0.0
        candidates = (set(sem) | {p for p, v in kw.items() if v > 0}) & eligible
        rows = []
        for path in candidates:
            meta = self.registry[path]
            chunks = sem.get(path, [])
            sims = sorted((c["sim"] for c in chunks), reverse=True)
            raw = (0.6 * sims[0] + 0.4 * float(np.mean(sims[:3]))) if sims else 0.0
            s_sem = float(np.clip((raw - floor) / (ceil - floor), 0, 1))
            s_kw = (kw.get(path, 0.0) / kw_max) if kw_max else 0.0
            s_skill, matched, missing = self._skill_coverage(meta, jd)
            s_exp = min(1.0, meta["experience_years"] / jd["min_years"]) if jd["min_years"] else 1.0
            if mode == "semantic":
                score = s_sem
            elif mode == "keyword":
                score = s_kw
            else:
                score = (WEIGHTS["semantic"] * s_sem + WEIGHTS["keyword"] * s_kw
                         + WEIGHTS["skills"] * s_skill + WEIGHTS["experience"] * s_exp)
            rows.append((score, path, meta, chunks, matched, missing,
                         {"semantic": round(s_sem, 3), "keyword": round(s_kw, 3),
                          "skill_coverage": round(s_skill, 3), "experience_fit": round(s_exp, 3),
                          "raw_cosine": round(raw, 3)}))
        rows.sort(key=lambda r: (-r[0], r[2]["candidate_name"]))

        top = []
        for score, path, meta, chunks, matched, missing, breakdown in rows[:k]:
            score100 = int(round(score * 100))
            best = max((c["sim"] for c in chunks), default=0)
            sections = []
            for c in sorted(chunks, key=lambda c: -c["sim"]):
                sec = c["section"].title()
                if c["sim"] >= best - 0.08 and sec not in sections and sec != "Header":
                    sections.append(sec)
            sections = sections[:3]
            reasoning = None
            if use_llm and llm_available():
                reasoning = generate_reasoning(jd_text, {
                    "score": score100, "years": meta["experience_years"], "matched_skills": matched,
                    "missing_skills": missing, "sections": sections})
            reasoning = reasoning or self._template_reasoning(score100, meta, jd, matched, missing, sections)
            top.append({
                "candidate_name": meta["candidate_name"],
                "resume_path": path,
                "match_score": score100,
                "matched_skills": matched,
                "relevant_excerpts": self._excerpts(chunks, matched or jd["skills"]),
                "reasoning": reasoning,
                # extras beyond the required schema
                "missing_skills": missing,
                "experience_years": meta["experience_years"],
                "matched_sections": sections,
                "score_breakdown": breakdown,
            })
        t_end = time.perf_counter()
        return {
            "job_description": jd_text.strip(),
            "top_matches": top,
            "parsed_requirements": {"must_have": jd["must_have"], "min_years": jd["min_years"],
                                    "required_skills": jd["required"], "nice_to_have": jd["nice_to_have"]},
            "filtered_out": excluded,
            "retrieval": {"mode": mode, "embedder": self.rag.embedder.name, "k": k,
                          "candidates_scored": len(rows), "filters_applied": bool(apply_filters),
                          "latency_ms": {"semantic": round((t_sem - t0) * 1000, 1),
                                         "keyword": round((t_kw - t_sem) * 1000, 1),
                                         "total": round((t_end - t0) * 1000, 1)}},
        }


def main():
    ap = argparse.ArgumentParser(description="Match resumes to a job description")
    ap.add_argument("--jd", required=True, help="path to a JD file, or the JD text itself")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--mode", choices=["hybrid", "semantic", "keyword"], default="hybrid")
    ap.add_argument("--no-filter", action="store_true", help="disable must-have filtering")
    ap.add_argument("--llm", action="store_true", help="LLM-written reasoning if an API key is set")
    ap.add_argument("--out", help="write JSON here")
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()
    from .embeddings import get_embedder
    jd_text = Path(args.jd).read_text() if Path(args.jd).exists() else args.jd
    matcher = JobMatcher(ResumeRAG(embedder=get_embedder(args.provider)))
    result = matcher.match(jd_text, k=args.k, mode=args.mode, apply_filters=not args.no_filter, use_llm=args.llm)
    out = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out)
        print(f"wrote {args.out}")
    print(out)


if __name__ == "__main__":
    main()
