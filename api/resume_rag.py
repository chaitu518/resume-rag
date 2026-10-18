"""Part A - Resume RAG setup.

Pipeline:  load (file_tools) -> extract metadata -> section-aware chunking
           -> embed -> store in ChromaDB with filterable metadata.

CLI:
    python -m api.resume_rag --resumes data/resumes --reset
    python -m api.resume_rag --stats
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chromadb

from .chunker import chunk_resume, naive_chunk
from .embeddings import BaseEmbedder, get_embedder
from .file_tools import load_documents
from .metadata import extract_metadata
from .skills_taxonomy import skill_key

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "vector_db"


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)[:50]


class ResumeRAG:
    def __init__(self, db_path: str | Path = DEFAULT_DB, embedder: BaseEmbedder | None = None,
                 chunk_size: int = 1000, chunk_overlap: int = 120, chunking: str = "section"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or get_embedder()
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
        self.chunking = chunking  # "section" (default) or "naive" fixed-size baseline
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection_name = "resumes__" + _slug(self.embedder.name) + ("" if chunking == "section" else "__" + chunking)
        self.registry_path = self.db_path / f"{self.collection_name}_registry.json"
        if self.embedder.needs_fit:
            self.embedder.load(self.db_path / self.collection_name)
        self.collection = self.client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine", "embedder": self.embedder.name})

    # ---------- ingestion ----------
    def ingest(self, resume_dir: str | Path, reset: bool = False, verbose: bool = True) -> dict:
        t0 = time.perf_counter()
        if reset:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                self.collection_name, metadata={"hnsw:space": "cosine", "embedder": self.embedder.name})
        docs = load_documents(resume_dir)
        t_load = time.perf_counter()

        registry, all_chunks = {}, []
        for doc in docs:
            rel = (Path(doc.path).resolve().relative_to(ROOT).as_posix() if Path(doc.path).resolve().is_relative_to(ROOT) else Path(doc.path).as_posix())
            meta = extract_metadata(doc.text, rel)
            meta["file_type"] = doc.file_type
            registry[rel] = meta
            chunks = (chunk_resume(doc.text, self.chunk_size, self.chunk_overlap) if self.chunking == "section"
                      else naive_chunk(doc.text, 500, 50))
            for ch in chunks:
                cm = {
                    "candidate_name": meta["candidate_name"],
                    "resume_path": rel,
                    "section": ch.section,
                    "chunk_index": ch.chunk_index,
                    "experience_years": float(meta["experience_years"]),
                    "highest_degree": meta["highest_degree"],
                    "skills": ", ".join(meta["skills"]),
                }
                # boolean flag per skill -> enables Chroma `where` pre-filtering
                for s in meta["skills"]:
                    cm[skill_key(s)] = True
                all_chunks.append((f"{_slug(rel)}::{ch.chunk_index}", ch.text, cm))
        t_chunk = time.perf_counter()

        texts = [c[1] for c in all_chunks]
        if self.embedder.needs_fit:
            self.embedder.fit(texts)
            self.embedder.save(self.db_path / self.collection_name)
        vectors = self.embedder.embed_documents(texts) if texts else []
        t_embed = time.perf_counter()

        for i in range(0, len(all_chunks), 256):
            batch = all_chunks[i:i + 256]
            self.collection.upsert(
                ids=[c[0] for c in batch], documents=[c[1] for c in batch],
                metadatas=[c[2] for c in batch], embeddings=[v.tolist() for v in vectors[i:i + 256]])
        t_store = time.perf_counter()

        existing = self.load_registry() if not reset else {}
        existing.update(registry)
        self.registry_path.write_text(json.dumps(existing, indent=2))
        stats = {
            "embedder": self.embedder.name, "collection": self.collection_name,
            "resumes": len(docs), "chunks": len(all_chunks),
            "timing_s": {"load": round(t_load - t0, 3), "metadata_and_chunk": round(t_chunk - t_load, 3),
                         "embed": round(t_embed - t_chunk, 3), "store": round(t_store - t_embed, 3),
                         "total": round(t_store - t0, 3)},
        }
        if verbose:
            print(json.dumps(stats, indent=2))
        return stats

    # ---------- accessors used by the matcher ----------
    def load_registry(self) -> dict:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())
        return {}

    def count(self) -> int:
        return self.collection.count()

    def all_chunks(self) -> dict:
        return self.collection.get(include=["documents", "metadatas"])

    def query(self, embedding, n_results: int = 50, where: dict | None = None) -> dict:
        n = min(n_results, max(1, self.count()))
        return self.collection.query(query_embeddings=[list(map(float, embedding))], n_results=n, where=where,
                                     include=["documents", "metadatas", "distances"])


def main():
    ap = argparse.ArgumentParser(description="Index resumes into ChromaDB")
    ap.add_argument("--resumes", default=str(ROOT / "data" / "resumes"))
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--provider", default=None, help="openai | cohere | huggingface | local | auto")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--stats", action="store_true", help="print indexed metadata and exit")
    args = ap.parse_args()
    rag = ResumeRAG(args.db, get_embedder(args.provider))
    if args.stats:
        reg = rag.load_registry()
        print(f"{rag.collection_name}: {rag.count()} chunks, {len(reg)} resumes")
        for path, m in reg.items():
            print(f"  {m['candidate_name']:<20} {m['experience_years']:>4} yrs  {m['highest_degree']:<10} {', '.join(m['skills'][:6])}")
        return
    rag.ingest(args.resumes, reset=args.reset)


if __name__ == "__main__":
    main()
