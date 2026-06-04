import asyncio
from api.anthropic_client import call
from generation.core import run

MODEL = "claude-opus-4-7"   

if __name__ == "__main__":
    asyncio.run(run(MODEL, call))
