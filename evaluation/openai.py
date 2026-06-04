import asyncio
import glob
import os

import aiohttp
from dotenv import load_dotenv

load_dotenv()

from evaluation.core import JUDGE_SYSTEM, run

API_KEY  = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://api.openai.com/v1/chat/completions"

_EVAL_SEM_OPENAI = None


async def call(session: aiohttp.ClientSession, model: str, prompt: str,
               temperature: float = 0) -> dict:
    global _EVAL_SEM_OPENAI
    if _EVAL_SEM_OPENAI is None:
        _EVAL_SEM_OPENAI = asyncio.Semaphore(80)  

    async with _EVAL_SEM_OPENAI:
        payload = {
            "model":       model,
            "messages":    [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {API_KEY}"}
        try:
            async with session.post(BASE_URL, headers=headers, json=payload) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", 10))
                    await asyncio.sleep(retry_after)
                    return await call(session, model, prompt, temperature)
                if resp.status != 200:
                    text = await resp.text()
                    return {"text": f"ERROR:{resp.status}:{text[:200]}", "input_tokens": 0, "output_tokens": 0}
                data = await resp.json()
                usage = data.get("usage", {})
                return {
                    "text":          data["choices"][0]["message"]["content"],
                    "input_tokens":  usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                }
        except Exception as e:
            return {"text": f"ERROR:exception:{str(e)[:200]}", "input_tokens": 0, "output_tokens": 0}


JUDGE_MODEL = "gpt-5.4-mini"

GEN_FILES = sorted(glob.glob("data/generations/*.jsonl"))

if __name__ == "__main__":
    if not GEN_FILES:
        print("No generations_*.jsonl files found. Run generation scripts first.")
    else:
        print(f"Judge     : {JUDGE_MODEL}")
        print(f"Gen files : {GEN_FILES}")
        asyncio.run(run(JUDGE_MODEL, call, GEN_FILES))