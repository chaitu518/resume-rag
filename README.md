# Resume RAG & Intelligent Matching Agent

An AI hiring assistant built in three milestones:

| Milestone | What it adds | Code |
|---|---|---|
| **1. LLM-Powered File System Assistant** | Tools to list, read (PDF / DOCX / TXT), search and write files | [`milestone1/fs_tools.py`](milestone1/fs_tools.py) |
| **2. Resume RAG & Job Matching Engine** | Section-aware chunking, ChromaDB, hybrid semantic + BM25 search, 0-100 match scores | [`api/`](api/) |
| **3. Intelligent Matching Agent** | LangGraph agent: conversation, 3-round screening, explanations, refinement | [`matching_agent.py`](matching_agent.py), [`agent/`](agent/) |

📄 **Milestone 3 full write-up:** [MILESTONE3.md](MILESTONE3.md)
🎬 **Demo video:** _add your link here_

![Milestone 3 state machine](docs/m3/state_machine.png)

---

## Quick start

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONUTF8 = "1"

python scripts/build_m3_corpus.py     # 100 resumes for Milestone 3
python run_demo.py                    # hands-free tour of the whole project (~6 min)
python chat_cli.py --trace            # chat with the agent yourself
```

**macOS / Linux**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/build_m3_corpus.py
python run_demo.py
python chat_cli.py --trace
```

Other entry points:

| Command | What it does |
|---|---|
| `streamlit run app_streamlit.py` | Web chat for the agent (http://localhost:8501) |
| `pytest -q tests/test_m3_agent.py` | 8 conversation flows + integration tests |
| `python scripts/evaluate_m3.py` | Agent vs Milestone 2 ranking quality |
| `python app.py` | Milestone 2 web UI (http://localhost:8000) |

No API key is needed. Everything runs offline with deterministic scores.

---

## Milestone 3 at a glance

**Try saying:**

- `Find me candidates with React and 3+ years experience`
- `Compare the top 3 matches side by side`
- `Why did John rank higher than Jane?`
- `Make Next.js a must-have and require 5+ years` (the agent re-ranks and explains each move)
- `tell me about #2` and `interview questions for #1`

**How it works:**

1. **Parse JD** with the Milestone 1 `read_file` tool.
2. **Extract requirements**: must-have vs nice-to-have, built on Milestone 2's JD parser.
3. **Search resumes** with the Milestone 2 hybrid RAG matcher over 100 resumes.
4. **Round 1** scores all 100 and keeps the top 10.
5. **Round 2** runs a deep analysis of the top 10: evidence, recency, seniority, per-skill years.
6. **Round 3** decides STRONG HIRE / HIRE / BORDERLINE / NO HIRE, with suggestions for borderline candidates.
7. **Report** is saved with the Milestone 1 `write_file` tool.
8. **Human feedback loop**: refine, compare, explain, interview questions, then back to step 3.

**Ranking quality** (top 10 from 100 resumes, 6 job descriptions):

| System | P@10 | R-Prec | nDCG@10 |
|---|---|---|---|
| Milestone 2 hybrid search | 0.733 | 0.835 | 0.970 |
| **Milestone 3 agent (final round)** | **0.767** | **1.000** | **1.000** |

---

## Project structure

```text
├── milestone1/
│   └── fs_tools.py           # Milestone 1 file tools (unchanged)
├── api/                      # Milestone 2 engine
│   ├── file_tools.py         # PDF / DOCX / TXT / MD loaders
│   ├── chunker.py            # section-aware chunking
│   ├── metadata.py           # name, skills, experience years, education
│   ├── skills_taxonomy.py    # canonical skills + aliases
│   ├── embeddings.py         # OpenAI / Cohere / HuggingFace / offline TF-IDF-LSA
│   ├── resume_rag.py         # ingestion -> ChromaDB
│   ├── job_matcher.py        # JD parsing, hybrid search, scoring
│   ├── evaluate.py           # retrieval metrics
│   ├── endpoints.py          # REST API routes
│   └── generate_dataset.py   # synthetic dataset + ground truth
├── agent/                    # Milestone 3 agent logic
│   ├── requirements.py       # must-have / nice-to-have extraction
│   ├── candidates.py         # candidate profiles (M1 read_file + M2 metadata)
│   ├── scoring.py            # rounds 1-3
│   ├── explain.py            # "why A > B", ranking-change diffs, comparisons
│   ├── nlu.py                # intent routing + refinement parsing
│   ├── report.py             # match reports
│   └── tools/                # LangChain tools: M1 files, M2 rag_search, M3 tools
├── matching_agent.py         # LangGraph state machine (Milestone 3 entry point)
├── chat_cli.py               # terminal chat
├── app_streamlit.py          # web chat
├── run_demo.py               # hands-free demo
├── app.py                    # Milestone 2 FastAPI app
├── data/
│   ├── resumes/              # 36 resumes (Milestone 2)
│   ├── resumes_m3/           # 100 resumes (Milestone 3)
│   ├── job_descriptions/     # 6 JDs
│   ├── ground_truth.json     # Milestone 2 relevance labels
│   └── ground_truth_m3.json  # Milestone 3 relevance labels
├── docs/m3/                  # state machine diagrams, demo script, conversation transcripts
├── tests/                    # pytest flows + scripted scenarios
├── scripts/                  # corpus builder, evaluation, diagram export, scenario runner
├── notebooks/                # Milestone 2 experiments
└── outputs/                  # match JSON, metrics, reports
```

---

## Milestone 2 details

Given a job description, the engine returns the top-10 matching resumes with a 0-100 score, matched skills, supporting excerpts and a reason. It uses a RAG pipeline (section-aware chunking → embeddings → ChromaDB) plus hybrid semantic + keyword search and must-have filtering.

### Running Milestone 2

```bash
python -m api.generate_dataset            # optional: dataset is already in data/
python -m api.resume_rag --reset          # index resumes
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
| `cohere` | embed-english-v3.0 | `COHERE_API_KEY`, `pip install cohere` |
| `huggingface` | BAAI/bge-small-en-v1.5 (override with `HF_EMBED_MODEL`) | `sentence-transformers`, internet once |
| `local` | TF-IDF (1-2 grams) → SVD-128 (latent semantic analysis) | nothing |
| `auto` (default) | first available of the above | |

Each provider gets its own Chroma collection, so you can index with several and compare them (notebook section 5.4). Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` and pass `--llm` for LLM-written reasoning. The default is a deterministic template, so results are reproducible.

### Part A: ingestion (`api/resume_rag.py`)

1. **Load** every file with `file_tools` (pypdf, python-docx, plain text), keeping line structure.
2. **Extract metadata:** name, title, email, skills (alias-normalised), and experience years. Date ranges are merged so overlapping jobs aren't double-counted, and *Present* is resolved against the reference date. Also per-skill years from role blocks, highest degree and field.
3. **Chunk by section:** headers such as *Work Experience*, *Employment History* and *Academic Background* map to canonical sections. Each chunk is prefixed `Name | Section` so it embeds with context. Long sections split on role boundaries with 120-character overlap.
4. **Embed and store** in a persistent ChromaDB collection (cosine HNSW). Chunk metadata holds the candidate, path, section, `experience_years`, degree, skills string, and a `skill_<name>: true` flag per skill for `where` filtering.

### Part B: matching (`api/job_matcher.py`)

1. **Parse the JD** clause by clause: all skills, must-haves (`N+ years X`, *Must have* blocks, *required*), nice-to-haves (*a plus*, *nice to have*), and minimum years.
2. **Must-have filter** pushed into ChromaDB, e.g. `{"$and": [{"skill_python": true}, {"experience_years": {"$gte": 5}}]}`. Excluded candidates are returned in `filtered_out` with reasons.
3. **Hybrid retrieval:** semantic (JD embedding vs chunks, top 200) combined with keyword (BM25 over chunks; the query is the JD skills and aliases, with must-haves doubled).
4. **Score (0-100)** = 100 × (0.40 semantic + 0.15 keyword + 0.35 skill coverage + 0.10 experience fit). Semantic = `0.6·max + 0.4·mean(top-3)` chunk similarity, calibrated per model. Skill coverage weights must-haves ×2, required ×1, nice-to-have ×0.5.
5. **Explain:** matched and missing skills, matched sections, up to 3 excerpts (experience bullets preferred), and reasoning.

### Output format

```json
{
  "job_description": "...",
  "top_matches": [{
    "candidate_name": "Priya Tanaka",
    "resume_path": "data/resumes/priya_tanaka.pdf",
    "match_score": 90,
    "matched_skills": ["Machine Learning", "NLP", "LLMs", "Python", "Deep Learning", "PyTorch", "AWS", "Docker"],
    "relevant_excerpts": ["Fine-tuned transformer models for NLP tasks such as entity extraction and intent classification"],
    "reasoning": "Strong match: 6.0 yrs experience (role asks for 5+); covers 8/11 JD skills; gaps: MLOps, TensorFlow",
    "missing_skills": ["MLOps", "TensorFlow"],
    "experience_years": 6.0,
    "matched_sections": ["Skills"],
    "score_breakdown": {"semantic": 0.959, "keyword": 1.0, "skill_coverage": 0.773, "experience_fit": 1.0}
  }],
  "parsed_requirements": {"must_have": {"Python": 5}, "min_years": 5},
  "filtered_out": [{"candidate_name": "...", "reasons": ["missing must-have skill: Python"]}],
  "retrieval": {"mode": "hybrid", "latency_ms": {"total": 28.0}}
}
```

### Dataset

36 synthetic resumes across 9 role families (ML, Java backend, Python backend, data engineering, DevOps, frontend, mobile, QA, data analyst), with 1-10 years of experience. They vary file format, section header wording and order, and skill synonyms. They also include hard distractors, such as Python backend developers against an ML JD, and juniors in the right family. `ground_truth.json` grades every resume per JD: 2 = right family and meets min years, 1 = right family but junior, 0 = other.

### Results (offline `local:tfidf-lsa-128` embeddings)

| Setting | R-Prec | R@10 | MRR | nDCG@10 | Meets years |
|---|---|---|---|---|---|
| Semantic only | 0.78 | 1.00 | 0.92 | 0.95 | 0.77 |
| Keyword only (BM25) | 0.78 | 1.00 | 0.92 | 0.96 | 0.73 |
| **Hybrid** | **0.94** | 1.00 | **1.00** | **0.995** | 0.82 |
| **Hybrid + must-have filter** | **1.00** | 1.00 | 1.00 | 0.91 ¹ | **1.00** |

¹ Lower by design: right-family juniors (grade 1) are excluded by the years filter.

| Paraphrased queries | P@3 | P@5 | MRR |
|---|---|---|---|
| Semantic | 0.75 | 0.60 | 0.90 |
| Keyword | 0.33 | 0.30 | 0.35 |
| Hybrid | 0.83 | 0.68 | 0.90 |

- **Score separation:** ideal candidates average about 91, and wrong-role candidates never exceed 45.
- **Metadata:** 100% name and degree accuracy, exact experience years, 99.7% skill recall.
- **Latency:** ingestion about 0.65 s for 36 resumes; queries about 11-16 ms p50 and under 40 ms p95.

### Limitations

- Synthetic resumes are cleaner than real ones (no multi-column layouts or tables).
- In Milestone 2, "5+ years Python" is enforced as total experience plus the skill being present. Milestone 3 also checks per-skill years in round 2.
- The offline embedder misses vocabulary it never saw (e.g. *JVM* ≈ Java). Use a neural provider for production.
