"""Section-aware chunking for resumes.

Resumes have a predictable shape (Summary, Experience, Education, Skills,
Projects ...). Splitting on those headers keeps each chunk semantically
coherent, so a JD asking for "ML experience" hits the Experience chunk,
not half of Experience glued to half of Education. Long sections are
split further on role / paragraph boundaries with a small overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# canonical section -> header variants seen in real resumes
SECTION_HEADERS = {
    "summary": ["summary", "professional summary", "profile", "objective", "about me", "career objective"],
    "experience": ["experience", "work experience", "professional experience", "employment history",
                   "work history", "career history"],
    "education": ["education", "academic background", "academics", "qualifications", "education & training"],
    "skills": ["skills", "technical skills", "core skills", "key skills", "core competencies", "tech stack"],
    "projects": ["projects", "key projects", "personal projects", "selected projects"],
    "certifications": ["certifications", "certificates", "licenses & certifications", "awards & certifications"],
}
_HEADER_LOOKUP = {v: k for k, vs in SECTION_HEADERS.items() for v in vs}
_HEADER_RE = re.compile(r"^\s*([A-Za-z &/]{3,40}?)\s*:?\s*$")


@dataclass
class Chunk:
    text: str
    section: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def detect_header(line: str) -> str | None:
    m = _HEADER_RE.match(line)
    if not m:
        return None
    key = m.group(1).strip().lower()
    return _HEADER_LOOKUP.get(key)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Return [(section_name, section_text)], header block first."""
    sections: list[tuple[str, list[str]]] = [("header", [])]
    for line in text.split("\n"):
        sec = detect_header(line)
        if sec:
            sections.append((sec, []))
        else:
            sections[-1][1].append(line)
    out = []
    for name, lines in sections:
        body = "\n".join(lines).strip()
        if body:
            out.append((name, body))
    return out


def _split_long(body: str, max_chars: int, overlap: int) -> list[str]:
    """Split on blank lines / role lines first, then hard-wrap if still too long."""
    if len(body) <= max_chars:
        return [body]
    blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if b.strip()]
    if len(blocks) == 1:  # no blank lines: split on lines instead
        blocks = [b for b in body.split("\n") if b.strip()]
    pieces, cur = [], ""
    for b in blocks:
        if cur and len(cur) + len(b) + 1 > max_chars:
            pieces.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + "\n" + b).strip()
        else:
            cur = (cur + "\n" + b).strip()
    if cur:
        pieces.append(cur)
    final = []
    for p in pieces:  # last resort hard wrap
        while len(p) > max_chars * 1.5:
            final.append(p[:max_chars])
            p = p[max_chars - overlap:]
        final.append(p)
    return final


def chunk_resume(text: str, max_chars: int = 1000, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    sections = split_sections(text)
    header = next((b for s, b in sections if s == "header"), "")
    name_line = header.split("\n")[0] if header else ""
    for section, body in sections:
        for piece in _split_long(body, max_chars, overlap):
            # Prefix with name + section so each chunk embeds with context
            prefix = f"{name_line} | {section.title()}\n" if section != "header" else ""
            chunks.append(Chunk(text=prefix + piece, section=section, chunk_index=len(chunks)))
    return chunks


def naive_chunk(text: str, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Fixed-size baseline used in the notebook for comparison."""
    out, i = [], 0
    while i < len(text):
        out.append(Chunk(text=text[i:i + size], section="unknown", chunk_index=len(out)))
        i += size - overlap
    return out
