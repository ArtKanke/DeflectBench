import asyncio
import json
import time                             
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from config import CLAIMS, PROMPTS, RUNS_PER_CONDITION, build_prompt

BATCH_SIZE = 50   



def model_slug(model: str) -> str:
    """Short filesystem-safe model identifier."""
    return (model.replace("/", "_")
                 .replace("-", "_")
                 .replace(".", "_"))[:30]

def make_gen_id(claim_id: str, prompt_id: str, slug: str, run: int) -> str:
    return f"{claim_id}__{prompt_id}__{slug}__r{run}"

def output_path(slug: str) -> str:
    return f"data/generations/generations_{slug}.jsonl"

def checkpoint_path(slug: str) -> str:
    return f"data/generations/checkpoint_{slug}.txt"



def load_completed(slug: str) -> set:
    p = Path(checkpoint_path(slug))
    if not p.exists():
        return set()
    return set(p.read_text().splitlines())

def mark_done(slug: str, gen_id: str):
    with open(checkpoint_path(slug), "a") as f:
        f.write(gen_id + "\n")

def write_record(slug: str, record: dict):
    with open(output_path(slug), "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")



async def run(model: str, api_caller):
    """
    api_caller signature: async (session, model, prompt, temperature) -> dict
    dict shape: {"text": str, "input_tokens": int, "output_tokens": int}
    """
    Path("data/generations").mkdir(parents=True, exist_ok=True)
    slug      = model_slug(model)
    completed = load_completed(slug)

    tasks = []
    for claim in CLAIMS:
        for prompt_cfg in PROMPTS:
            for run_num in range(1, RUNS_PER_CONDITION + 1):
                gen_id = make_gen_id(
                    claim["claim_id"], prompt_cfg["prompt_id"], slug, run_num
                )
                if gen_id not in completed:
                    tasks.append((gen_id, claim, prompt_cfg, run_num))

    total_conditions = len(CLAIMS) * len(PROMPTS) * RUNS_PER_CONDITION
    print(f"Model     : {model}")
    print(f"Output    : {output_path(slug)}")
    print(f"Total     : {total_conditions}  |  Done: {len(completed)}  |  Remaining: {len(tasks)}")

    if not tasks:
        print("Nothing to do — all generations complete.")
        return

    token_lock    = asyncio.Lock()
    total_in_tok  = 0
    total_out_tok = 0
    start_time    = time.time()        

    async with aiohttp.ClientSession() as session:

        async def process(gen_id, claim, prompt_cfg, run_num):
            nonlocal total_in_tok, total_out_tok

            prompt_text = build_prompt(prompt_cfg, claim)
            result      = await api_caller(session, model, prompt_text)

            async with token_lock:
                total_in_tok  += result["input_tokens"]
                total_out_tok += result["output_tokens"]

            record = {
                "gen_id":            gen_id,
                "claim_id":          claim["claim_id"],
                "claim_text":        claim["text"],
                "controversy_level": claim["level"],
                "context":           claim["context"],
                "domain":            claim["domain"],
                "fallacy":           prompt_cfg["fallacy"],
                "explicitness":      prompt_cfg["explicitness"],
                "frame":             prompt_cfg["frame"],
                "prompt_num":        prompt_cfg["prompt_num"],
                "prompt_id":         prompt_cfg["prompt_id"],
                "prompt_text":       prompt_text,
                "model":             model,
                "run":               run_num,
                "response":          result["text"],          
                "input_tokens":      result["input_tokens"], 
                "output_tokens":     result["output_tokens"], 
                "is_error":          int(result["text"].startswith("ERROR:")),  
                "timestamp":         datetime.now(timezone.utc).isoformat(),
            }
            write_record(slug, record)
            mark_done(slug, gen_id)

        done = 0
        for i in range(0, len(tasks), BATCH_SIZE):
            batch = tasks[i : i + BATCH_SIZE]
            await asyncio.gather(*[process(*t) for t in batch])
            done += len(batch)
            pct = done / len(tasks) * 100
            print(f"  [{pct:5.1f}%]  {done}/{len(tasks)} generated", flush=True)

    elapsed = time.time() - start_time
    print(f"\nFinished. Output → {output_path(slug)}")
    print(f"─────────────────────────────────────")
    print(f"Time elapsed  : {elapsed/60:.1f} min  ({elapsed:.0f}s)")
    print(f"Input tokens  : {total_in_tok:,}")
    print(f"Output tokens : {total_out_tok:,}")
    print(f"Total tokens  : {total_in_tok + total_out_tok:,}")