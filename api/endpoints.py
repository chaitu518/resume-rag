"""REST endpoints for the resume matcher (mounted under /api by app.py)."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .job_matcher import JobMatcher
from .resume_rag import ROOT, ResumeRAG

router = APIRouter()
_state: dict = {}


def get_matcher() -> JobMatcher:
    if "matcher" not in _state:
        rag = ResumeRAG()
        if rag.count() == 0:
            rag.ingest(ROOT / "data" / "resumes", reset=True, verbose=False)
        _state["matcher"] = JobMatcher(rag)
    return _state["matcher"]


class MatchRequest(BaseModel):
    job_description: str = Field(..., min_length=20)
    k: int = Field(10, ge=1, le=50)
    mode: str = Field("hybrid", pattern="^(hybrid|semantic|keyword)$")
    apply_filters: bool = True
    use_llm: bool = False


class IngestRequest(BaseModel):
    resume_dir: str = "data/resumes"
    reset: bool = True


@router.get("/health")
def health():
    m = get_matcher()
    return {"status": "ok", "embedder": m.rag.embedder.name, "chunks": m.rag.count(), "resumes": len(m.registry)}


@router.get("/resumes")
def list_resumes():
    return sorted(get_matcher().registry.values(), key=lambda r: r["candidate_name"])


@router.get("/job-descriptions")
def sample_jds():
    jd_dir = ROOT / "data" / "job_descriptions"
    return [{"id": p.stem, "text": p.read_text()} for p in sorted(jd_dir.glob("*.txt"))]


@router.post("/ingest")
def ingest(req: IngestRequest):
    path = (ROOT / req.resume_dir).resolve()
    if not path.is_relative_to(ROOT) or not path.exists():
        raise HTTPException(400, "resume_dir must be an existing folder inside the project")
    rag = ResumeRAG()
    stats = rag.ingest(path, reset=req.reset, verbose=False)
    _state["matcher"] = JobMatcher(rag)
    return stats


@router.post("/match")
def match(req: MatchRequest):
    return get_matcher().match(req.job_description, k=req.k, mode=req.mode,
                               apply_filters=req.apply_filters, use_llm=req.use_llm)
