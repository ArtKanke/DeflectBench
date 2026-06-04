# DeflectBench

Paper: "DeflectBench: A Benchmark for Evaluating Rhetorical Fallacy Generation in
LLMs"

A benchmark for measuring LLM compliance with requests to generate rhetorical fallacies (whataboutism, ad hominem, and red herring) and for quantifying how prompt framing influences refusal behavior.


## Dataset summary

- 23,990 valid generations
- 4 frontier models x 80 claims x 15 prompt templates x 5 runs
- 7.46M generated tokens (9.72M including prompts)
- Two LLMs judges score 8 rubric fields per generation; 383,768 binary labels
- 9 domains, 4 controversy levels, two geopolitical contexts


## Repo layout

    config.py        80 claims, 15 prompt templates, fallacy definitions
    api/             one client per API
    generation/      response generation; shared core loop + one entrypoint per model
    evaluation/      judge scoring; shared core loop + one entrypoint per model
    analysis/        analysis.py, stats_utils.py (bootstrap CIs, AC1, Cohen's h, TOST)
    data/
      claims/        the 80 claims (claims.jsonl)
      generations/   raw model outputs (generations_<model>.jsonl)
      evals/         judge labels (evals_<model>.jsonl)

## Install

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env   # then add your API keys

## Reproduce the paper (no API keys required)

Every table regenerates from the released data/ tree:

    make reproduce

or individually:

    python analysis/analysis.py
    python analysis/stats_utils.py --data-dir data


## Adding a model

The pipeline uses a API adapter pattern, so a new model is two small files.

1. If the API is new, add api/<provider>_client.py exposing an async call:

        async def call(session, model, prompt, temperature=1.0) -> dict
        # returns {"text": str, "input_tokens": int, "output_tokens": int}

2. Add generation/<name>.py:

        import asyncio
        from api.<provider>_client import call
        from generation.core import run
        MODEL = "<model-id>"
        if __name__ == "__main__":
            asyncio.run(run(MODEL, call))

3. Run python -m generation.<name>. A new judge follows the same pattern under evaluation/.

## Data schema

generations_<model>.jsonl: one row per generation with gen_id, claim_id, claim_text,
controversy_level, context, domain, fallacy, explicitness, frame, prompt_id, prompt_text,
model, run, response, input_tokens, output_tokens.

evals_<judge>.jsonl: one row per (generation, judge) with the eight rubric fields
(refusal, soft_refusal, WA_present, AH_present, RH_present, any_fallacy_present,
compliance_clean, fallacy_labeled) and parse_error.

## Dataset on Hugging Face

TBD.

## Citation

If you use DeflectBench in your work, please kindly cite it using the BibTeX entry below:

TBD.

