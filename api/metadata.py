"""Extract structured fields from resume text: name, skills, years, education."""
from __future__ import annotations

import re
from datetime import date

from .chunker import split_sections
from .skills_taxonomy import extract_skills

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
TODAY = date(2026, 10, 1)  # dataset reference date; override with set_today()

_DATE = r"(?:(?P<{p}m>[A-Za-z]{{3,9}})\.?\s+)?(?P<{p}y>(?:19|20)\d\d)"
_RANGE_RE = re.compile(
    _DATE.format(p="s") + r"\s*(?:-|–|—|to)\s*(?:" + _DATE.format(p="e") + r"|(?P<present>present|current|now|till date))",
    re.I,
)
_STATED_YEARS_RE = re.compile(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years|yrs)", re.I)
DEGREE_PATTERNS = [
    ("PhD", r"\b(ph\.?d|doctorate)\b"),
    ("Master's", r"\b(m\.?tech|m\.?s\.?|master'?s|m\.?sc|mba|m\.?e\.?|mca)\b"),
    ("Bachelor's", r"\b(b\.?tech|b\.?e\.?|b\.?s\.?|bachelor'?s|b\.?sc|bca|b\.?com|b\.?des)\b"),
    ("Diploma", r"\bdiploma\b"),
]


def set_today(d: date):
    global TODAY
    TODAY = d


def _to_month_index(month_str: str | None, year: str, end: bool) -> int:
    m = MONTHS.get((month_str or "").lower()[:3], 12 if end else 1)
    # end months are inclusive: "Jan 2020 - Dec 2020" is 12 months
    return int(year) * 12 + m + (1 if end else 0)


def experience_years(experience_text: str) -> float:
    """Sum non-overlapping date ranges in the experience section."""
    intervals = []
    for mt in _RANGE_RE.finditer(experience_text):
        start = _to_month_index(mt.group("sm"), mt.group("sy"), end=False)
        if mt.group("present"):
            end = TODAY.year * 12 + TODAY.month + 1
        else:
            end = _to_month_index(mt.group("em"), mt.group("ey"), end=True)
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return 0.0
    intervals.sort()
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    months = sum(e - s for s, e in merged)
    return round(months / 12, 1)


def skill_years(experience_text: str) -> dict[str, float]:
    """Approximate years per skill: sum durations of roles whose block mentions it."""
    from .skills_taxonomy import extract_skills as _skills
    matches = list(_RANGE_RE.finditer(experience_text))
    totals: dict[str, float] = {}
    for i, mt in enumerate(matches):
        # role block = title line just before the date + bullets until the next date range
        block_start = experience_text.rfind("\n", 0, max(0, mt.start() - 1)) + 1
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(experience_text)
        block = experience_text[block_start:block_end]
        start = _to_month_index(mt.group("sm"), mt.group("sy"), end=False)
        end = (TODAY.year * 12 + TODAY.month + 1) if mt.group("present") else _to_month_index(mt.group("em"), mt.group("ey"), True)
        dur = max(0, end - start) / 12
        for s in _skills(block):
            totals[s] = round(totals.get(s, 0) + dur, 1)
    return totals


def extract_name(text: str) -> str:
    for line in text.split("\n")[:5]:
        line = line.strip()
        if not line or "@" in line or re.search(r"\d{3}", line):
            continue
        words = line.split()
        if 1 < len(words) <= 4 and all(w[0].isupper() for w in words if w[0].isalpha()):
            return line
    return "Unknown"


def extract_education(text: str) -> dict:
    sections = dict(split_sections(text))
    edu = sections.get("education", text)
    highest, field_ = None, None
    for label, pat in DEGREE_PATTERNS:
        if re.search(pat, edu, re.I):
            highest = label
            break
    m = re.search(r"\b(?:in|of)\s+([A-Z][A-Za-z &]+?)(?:,|\n|\(|\s-\s|$)", edu)
    if m:
        field_ = m.group(1).strip()
    first_line = next((l.strip() for l in edu.split("\n") if l.strip()), "")
    return {"highest_degree": highest or "Unknown", "field": field_ or "", "education_summary": first_line[:200]}


def extract_metadata(text: str, path: str) -> dict:
    sections = dict(split_sections(text))
    exp_text = sections.get("experience", "")
    years = experience_years(exp_text)
    stated = [float(x) for x in _STATED_YEARS_RE.findall(sections.get("summary", "") + sections.get("header", ""))]
    if years == 0 and stated:
        years = max(stated)
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    skills = extract_skills(text)
    edu = extract_education(text)
    title_line = sections.get("header", "").split("\n")
    return {
        "candidate_name": extract_name(text),
        "resume_path": path,
        "email": email.group(0) if email else "",
        "current_title": title_line[1].strip() if len(title_line) > 1 and "@" not in title_line[1] else "",
        "skills": skills,
        "experience_years": years,
        "skill_years": skill_years(exp_text),
        "stated_years": max(stated) if stated else None,
        **edu,
    }
