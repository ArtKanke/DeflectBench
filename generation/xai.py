import asyncio
from api.xai_client import call
from generation.core import run

MODEL = "grok-4.3"   

if __name__ == "__main__":
    asyncio.run(run(MODEL, call))
