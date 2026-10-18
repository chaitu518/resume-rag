"""Optional LLM-written match reasoning.

If ANTHROPIC_API_KEY or OPENAI_API_KEY is set and use_llm=True, the matcher
asks an LLM to turn the structured match evidence into a 1-2 sentence
explanation. Otherwise it falls back to the deterministic template, so the
system works fully offline and results stay reproducible for evaluation.
"""
from __future__ import annotations

import json
import os

PROMPT = """You are helping a recruiter. Using ONLY the evidence below, write 1-2 sentences explaining
why this candidate does or does not fit the job. Mention concrete skills, years and which resume sections matched.
Do not invent facts.

Job summary: {jd}
Evidence: {evidence}"""


def llm_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))


def generate_reasoning(jd_text: str, evidence: dict, max_tokens: int = 120) -> str | None:
    prompt = PROMPT.format(jd=jd_text[:1200], evidence=json.dumps(evidence)[:2000])
    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
                                         max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
            return msg.content[0].text.strip()
        if os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            r = OpenAI().chat.completions.create(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                                                 max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content.strip()
    except Exception as e:  # never let reasoning break matching
        print(f"[llm] reasoning failed, using template: {e}")
    return None
