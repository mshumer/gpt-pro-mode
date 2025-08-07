import os
import time
from typing import List
import concurrent.futures as cf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

MODEL = "gpt-5"
MAX_OUTPUT_TOKENS = 30000          # cap per call
MAX_WORKERS = 16                   # limit fanout

app = FastAPI(title="Pro Mode (OpenAI Responses API, GPT-5)")


# ---------- Pydantic Schemas ----------
class ProModeRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    num_gens: int = Field(..., ge=1, le=32)  # keep sane upper bound


class ProModeResponse(BaseModel):
    final: str
    candidates: List[str]


# ---------- Helpers ----------
def _extract_text(resp) -> str:
    """Robustly extract text from a Responses API result."""
    # Preferred: SDK convenience
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt

    # Fallback: walk the structured output
    parts: List[str] = []
    output = getattr(resp, "output", None)
    if not output and isinstance(resp, dict):
        output = resp.get("output")

    if output:
        for item in output:
            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")
            if not content:
                continue
            for c in content:
                ctype = getattr(c, "type", None) or (isinstance(c, dict) and c.get("type"))
                if ctype in ("output_text", "text"):
                    text = getattr(c, "text", None) or (isinstance(c, dict) and c.get("text"))
                    if text:
                        parts.append(text)
    return "".join(parts).strip()


def _one_completion(api_key: str, prompt: str, temperature: float) -> str:
    """Single non-streaming completion with simple retry/backoff."""
    delay = 0.5
    for attempt in range(3):
        try:
            client = OpenAI(api_key=api_key)  # per-thread client for safety
            resp = client.responses.create(
                model=MODEL,
                input=prompt,
                temperature=temperature,       # 0.9 for candidates; 0.2 for synthesis
                top_p=1,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            return _extract_text(resp)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(delay)
            delay *= 2
    return ""


def _build_synthesis_inputs(candidates: List[str]) -> tuple[str, str]:
    numbered = "\n\n".join(
        f"<cand {i+1}>\n{txt}\n</cand {i+1}>" for i, txt in enumerate(candidates)
    )
    instructions = (
        "You are an expert editor. Synthesize ONE best answer from the candidate "
        "answers provided, merging strengths, correcting errors, and removing repetition. "
        "Do not mention the candidates or the synthesis process. Be decisive and clear."
    )
    user = (
        f"You are given {len(candidates)} candidate answers delimited by <cand i> tags.\n\n"
        f"{numbered}\n\nReturn the single best final answer."
    )
    return instructions, user


def _pro_mode(api_key: str, prompt: str, n_runs: int) -> ProModeResponse:
    # 1) Fan out candidates at high T
    num_workers = min(n_runs, MAX_WORKERS)
    candidates: List[str] = [""] * n_runs
    with cf.ThreadPoolExecutor(max_workers=num_workers) as ex:
        fut_to_idx = {
            ex.submit(_one_completion, api_key, prompt, 0.9): i
            for i in range(n_runs)
        }
        for fut in cf.as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            candidates[i] = fut.result()

    # Filter empty/failed
    filtered = [c for c in candidates if c and c.strip()]
    if not filtered:
        raise HTTPException(status_code=503, detail="All candidate generations failed.")

    # 2) Synthesis pass at low T using Responses API `instructions`
    client = OpenAI(api_key=api_key)
    instructions, user = _build_synthesis_inputs(filtered)
    final_resp = client.responses.create(
        model=MODEL,
        instructions=instructions,  # acts like a system message
        input=user,
        temperature=0.2,
        top_p=1,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    final_text = _extract_text(final_resp)

    return ProModeResponse(final=final_text, candidates=candidates)


# ---------- Routes ----------
@app.post("/pro-mode", response_model=ProModeResponse)
def pro_mode_endpoint(body: ProModeRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not set in environment."
        )
    try:
        return _pro_mode(api_key=api_key, prompt=body.prompt, n_runs=body.num_gens)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run(
        "main:app",                 # use import string for clean reload
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True                 # dev only
    )
