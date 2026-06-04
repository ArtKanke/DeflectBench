import asyncio
import glob
import os

import aiohttp
from dotenv import load_dotenv

load_dotenv()

from evaluation.core import JUDGE_SYSTEM, run

API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
BASE_URL = "https://api.anthropic.com/v1/messages"
_MAX_TOKENS = 512   

_EVAL_SEM_ANTHROPIC = None


async def call(session: aiohttp.ClientSession, model: str, prompt: str,
               temperature: float = 0) -> dict:
    global _EVAL_SEM_ANTHROPIC
    if _EVAL_SEM_ANTHROPIC is None:
        _EVAL_SEM_ANTHROPIC = asyncio.Semaphore(20)   

    async with _EVAL_SEM_ANTHROPIC:
        payload = {
            "model":      model,
            "max_tokens": _MAX_TOKENS,
            "system":     JUDGE_SYSTEM,
            "messages":   [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key":         API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        }
        try:
            async with session.post(BASE_URL, headers=headers, json=payload) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", 20))
                    await asyncio.sleep(retry_after)
                    return await call(session, model, prompt, temperature)
                if resp.status != 200:
                    text = await resp.text()
                    return {"text": f"ERROR:{resp.status}:{text[:200]}", "input_tokens": 0, "output_tokens": 0}
                data = await resp.json()
                usage = data.get("usage", {})
                return {
                    "text":          data["content"][0]["text"],
                    "input_tokens":  usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                }
        except Exception as e:
            return {"text": f"ERROR:exception:{str(e)[:200]}", "input_tokens": 0, "output_tokens": 0}


JUDGE_MODEL = "claude-haiku-4-5-20251001"

GEN_FILES = sorted(glob.glob("data/generations/*.jsonl"))

if __name__ == "__main__":
    if not GEN_FILES:
        print("No generations_*.jsonl files found. Run generation scripts first.")
    else:
        print(f"Judge     : {JUDGE_MODEL}")
        print(f"Gen files : {GEN_FILES}")
        asyncio.run(run(JUDGE_MODEL, call, GEN_FILES))