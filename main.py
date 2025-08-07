import os
import time
from typing import List, Tuple
import concurrent.futures as cf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

MODEL = "gpt-5"
MAX_OUTPUT_TOKENS = 30000
MAX_WORKERS = 100
MAX_GENS = 100
TOURNAMENT_THRESHOLD = 20
GROUP_SIZE = 10

app = FastAPI(title="Pro Mode (OpenAI Responses API, GPT-5)")

# ---------- Schemas ----------
class ProModeRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    num_gens: int = Field(..., ge=1, le=MAX_GENS)

class ProModeResponse(BaseModel):
    final: str
    candidates: List[str]

# ---------- Helpers ----------
def _extract_text(resp) -> str:
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt
    parts: List[str] = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) in ("output_text", "text"):
                parts.append(getattr(c, "text", ""))
    return "".join(parts).strip()

def _one_completion(api_key: str, prompt: str, temperature: float) -> str:
    delay = 0.5
    for attempt in range(3):
        try:
            client = OpenAI(api_key=api_key)  # per-thread client
            resp = client.responses.create(
                model=MODEL,
                input=prompt,
                temperature=temperature,
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

def _build_synthesis_io(candidates: List[str]) -> Tuple[str, str]:
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

def _synthesize(client: OpenAI, candidates: List[str]) -> str:
    instructions, user = _build_synthesis_io(candidates)
    resp = client.responses.create(
        model=MODEL,
        instructions=instructions,
        input=user,
        temperature=0.2,
        top_p=1,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return _extract_text(resp)

def _chunk(lst: List[str], size: int) -> List[List[str]]:
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def _fanout_candidates(api_key: str, prompt: str, n_runs: int, temp: float = 0.9) -> List[str]:
    num_workers = min(n_runs, MAX_WORKERS)
    results: List[str] = [""] * n_runs
    with cf.ThreadPoolExecutor(max_workers=num_workers) as ex:
        fut_to_idx = {
            ex.submit(_one_completion, api_key, prompt, temp): i
            for i in range(n_runs)
        }
        for fut in cf.as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            results[i] = fut.result()
    return results

def _pro_mode_simple(api_key: str, prompt: str, n_runs: int) -> ProModeResponse:
    candidates = _fanout_candidates(api_key, prompt, n_runs, temp=0.9)
    filtered = [c for c in candidates if c and c.strip()]
    if not filtered:
        raise HTTPException(status_code=503, detail="All candidate generations failed.")
    client = OpenAI(api_key=api_key)
    final_text = _synthesize(client, filtered)
    return ProModeResponse(final=final_text, candidates=candidates)

def _pro_mode_tournament(api_key: str, prompt: str, n_runs: int) -> ProModeResponse:
    # Round 1: fan out all candidates
    candidates = _fanout_candidates(api_key, prompt, n_runs, temp=0.9)
    filtered = [c for c in candidates if c and c.strip()]
    if not filtered:
        raise HTTPException(status_code=503, detail="All candidate generations failed.")

    # Group into chunks of 10 and synth each group (in parallel)
    groups = _chunk(filtered, GROUP_SIZE)
    client = OpenAI(api_key=api_key)

    def synth_group(group: List[str]) -> str:
        return _synthesize(client, group)

    with cf.ThreadPoolExecutor(max_workers=min(len(groups), MAX_WORKERS)) as ex:
        group_futures = [ex.submit(synth_group, g) for g in groups]
        group_winners = [f.result() for f in group_futures]

    # Final: synth across group winners
    final_text = _synthesize(client, group_winners)
    return ProModeResponse(final=final_text, candidates=candidates)

def _pro_mode(api_key: str, prompt: str, n_runs: int) -> ProModeResponse:
    if n_runs > TOURNAMENT_THRESHOLD:
        return _pro_mode_tournament(api_key, prompt, n_runs)
    else:
        return _pro_mode_simple(api_key, prompt, n_runs)

# ---------- Routes ----------
@app.post("/pro-mode", response_model=ProModeResponse)
def pro_mode_endpoint(body: ProModeRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment.")
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
