# Resume RAG & Job Matching Engine

Given a job description, return the top-10 matching resumes with a 0-100 score, matched skills, supporting excerpts
and a reason, using a RAG pipeline (section-aware chunking → embeddings → ChromaDB) plus hybrid semantic + keyword
search and must-have filtering.

```
resume-rag/
├── api/
│   ├── file_tools.py        # Milestone 1 loaders: PDF / DOCX / TXT / MD
│   ├── chunker.py           # section-aware chunking (+ naive baseline for ablation)
│   ├── metadata.py          # name, skills, experience years, education extraction
│   ├── skills_taxonomy.py   # canonical skills + aliases (ML, k8s, sklearn ...)
│   ├── embeddings.py        # OpenAI / Cohere / HuggingFace / offline TF-IDF-LSA
│   ├── resume_rag.py        # PART A: ingestion pipeline → ChromaDB
│   ├── job_matcher.py       # PART B: JD parsing, filters, hybrid search, scoring, JSON
│   ├── llm.py               # optional LLM-written reasoning
│   ├── evaluate.py          # retrieval accuracy, metadata accuracy, latency
│   ├── endpoints.py         # REST API routes
│   └── generate_dataset.py  # builds the dataset + ground truth
├── app.py                   # FastAPI app: UI at /, Swagger at /docs
├── data/
│   ├── resumes/             # 36 resumes (18 PDF, 9 DOCX, 9 TXT)
│   ├── job_descriptions/    # 6 JDs
│   └── ground_truth.json    # graded relevance labels + 8 paraphrased queries
├── notebooks/resume_rag_experiments.ipynb   # experiments, metrics, charts (executed)
├── outputs/                 # match JSON per JD + metrics_summary.json
├── ui/index.html            # single-page matcher UI
├── vector_db/               # persisted ChromaDB (created on first run)
├── DEMO_SCRIPT.md           # 3-4 minute video script
└── requirements.txt
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m api.generate_dataset            # optional: dataset is already in data/
python -m api.resume_rag --reset          # Part A: index resumes
python -m api.resume_rag --stats          # show extracted metadata

python -m api.job_matcher --jd data/job_descriptions/jd01_senior_ml_engineer.txt --out outputs/jd01.json
python -m api.job_matcher --jd "keep our container clusters healthy and write IaC" --mode semantic --no-filter --k 5

python app.py                             # http://localhost:8000
jupyter notebook notebooks/resume_rag_experiments.ipynb
```

### Choosing the embedding model

| `EMBEDDING_PROVIDER` | Model | Needs |
|---|---|---|
| `openai` | text-embedding-3-small | `OPENAI_API_KEY`, `pip install openai` |
| `cohere` | embed-english-v3.0 (document/query input types) | `COHERE_API_KEY`, `pip install cohere` |
| `huggingface` | BAAI/bge-small-en-v1.5 (override with `HF_EMBED_MODEL`) | `sentence-transformers`, internet once |
| `local` | TF-IDF (1-2 grams) → SVD-128 (latent semantic analysis) | nothing |
| `auto` (default) | first available of the above | |

Each provider gets its own Chroma collection, so you can index with several and compare (notebook section 5.4).
Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` and pass `--llm` to get LLM-written reasoning; the default is a
deterministic template so results are reproducible.

## How it works

### Part A - `api/resume_rag.py`
1. **Load** every file with `file_tools` (pypdf, python-docx, plain text), keeping line structure.
2. **Extract metadata**: name, title, email, skills (alias-normalised), experience years (date ranges merged so
   overlapping jobs are not double-counted; *Present* resolved against the reference date), per-skill years from
   role blocks, highest degree and field.
3. **Chunk by section**: headers such as *Work Experience / Employment History / Academic Background* map to canonical
   sections. Each chunk is prefixed `Name | Section` so it embeds with context; long sections split on role
   boundaries with 120-char overlap.
4. **Embed and store** in a persistent ChromaDB collection (cosine HNSW). Chunk metadata: candidate, path, section,
   `experience_years`, degree, skills string, and a `skill_<name>: true` flag per skill for `where` filtering.

### Part B - `api/job_matcher.py`
1. **Parse the JD** clause by clause: all skills, must-haves (`N+ years X`, *Must have* blocks, *required*,
   *must-have*), nice-to-haves (*a plus*, *nice to have*), minimum years.
2. **Must-have filter** pushed into ChromaDB: `{"$and": [{"skill_python": true}, {"experience_years": {"$gte": 5}}]}`.
   Excluded candidates are returned in `filtered_out` with reasons.
3. **Hybrid retrieval**: semantic (JD embedding vs chunks, top 200) ∪ keyword (BM25 over chunks; query = JD skills
   and aliases, must-haves doubled).
4. **Score (0-100)** = 100 × (0.40 semantic + 0.15 keyword + 0.35 skill coverage + 0.10 experience fit).
   Semantic = `0.6·max + 0.4·mean(top-3)` chunk similarity, calibrated per model. Skill coverage weights must-haves ×2,
   required ×1, nice-to-have ×0.5.
5. **Explain**: matched and missing skills, matched sections, up to 3 excerpts (experience bullets preferred), reasoning.

### Output (required fields plus extras)
```json
{
  "job_description": "...",
  "top_matches": [{
    "candidate_name": "Priya Tanaka",
    "resume_path": "data/resumes/priya_tanaka.pdf",
    "match_score": 90,
    "matched_skills": ["Machine Learning", "NLP", "LLMs", "Python", "Deep Learning", "PyTorch", "AWS", "Docker"],
    "relevant_excerpts": ["Fine-tuned transformer models for NLP tasks such as entity extraction and intent classification", "..."],
    "reasoning": "Strong match: 6.0 yrs experience (role asks for 5+); covers 8/11 JD skills ...; gaps: MLOps, TensorFlow ...",
    "missing_skills": ["MLOps", "TensorFlow"], "experience_years": 6.0, "matched_sections": ["Skills"],
    "score_breakdown": {"semantic": 0.959, "keyword": 1.0, "skill_coverage": 0.773, "experience_fit": 1.0}
  }],
  "parsed_requirements": {"must_have": {"Python": 5}, "min_years": 5, "...": "..."},
  "filtered_out": [{"candidate_name": "...", "reasons": ["missing must-have skill: Python"]}],
  "retrieval": {"mode": "hybrid", "latency_ms": {"total": 28.0}}
}
```

## Dataset
36 synthetic resumes across 9 role families (ML, Java backend, Python backend, data engineering, DevOps, frontend,
mobile, QA, data analyst) with 1-10 years of experience. They vary file format, section header wording and order, and
skill synonyms, and include hard distractors (Python backend devs vs an ML JD; juniors in the right family).
`ground_truth.json` grades every resume per JD: 2 = right family and meets min years, 1 = right family but junior, 0 = other.

## Results (offline `local:tfidf-lsa-128` embeddings; full detail in the notebook)

| Setting | R-Prec | R@10 | MRR | nDCG@10 | meets years |
|---|---|---|---|---|---|
| Semantic only | 0.78 | 1.00 | 0.92 | 0.95 | 0.77 |
| Keyword only (BM25) | 0.78 | 1.00 | 0.92 | 0.96 | 0.73 |
| **Hybrid** | **0.94** | 1.00 | **1.00** | **0.995** | 0.82 |
| **Hybrid + must-have filter** | **1.00** | 1.00 | 1.00 | 0.91* | **1.00** |

\* lower by design: right-family juniors (grade 1) are excluded by the years filter.

| Paraphrased queries | P@3 | P@5 | MRR |
|---|---|---|---|
| Semantic | 0.75 | 0.60 | 0.90 |
| Keyword | 0.33 | 0.30 | 0.35 |
| Hybrid | 0.83 | 0.68 | 0.90 |

* Score separation: ideal candidates average ≈ 91, wrong-role candidates never exceed 45.
* Metadata: 100% name/degree accuracy, experience years exact, 99.7% skill recall.
* Latency: ingestion ≈ 0.65 s for 36 resumes; query ≈ 11-16 ms p50 / < 40 ms p95.

## Limitations
* Synthetic resumes are cleaner than real ones (no multi-column layouts or tables).
* "5+ years Python" is enforced as total experience + skill present; per-skill years are extracted but not enforced.
* The offline embedder misses vocabulary it never saw (e.g. *JVM* ≈ Java); use a neural provider for production.
