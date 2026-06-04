from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd

SCORE_FIELDS = [
    "refusal", "soft_refusal", "WA_present", "AH_present", "RH_present",
    "any_fallacy_present", "compliance_clean", "fallacy_labeled",
]

def load_consensus(data_dir: str = "data") -> pd.DataFrame:
    gen_paths = sorted(glob.glob(str(Path(data_dir) / "generations" / "*.jsonl")))
    eval_paths = sorted(glob.glob(str(Path(data_dir) / "evals" / "*.jsonl")))
    if not gen_paths or not eval_paths:
        raise FileNotFoundError(
            f"No data under {data_dir}/generations and {data_dir}/evals. "
            "Pass the correct --data-dir."
        )
    gens = pd.concat([pd.read_json(f, lines=True) for f in gen_paths], ignore_index=True)
    gens = gens[~gens["response"].astype(str).str.startswith("ERROR:", na=False)].copy()

    evals = pd.concat([pd.read_json(f, lines=True) for f in eval_paths], ignore_index=True)
    evals = evals[evals["parse_error"] == 0]

    piv = evals.pivot_table(index="gen_id", columns="judge_model",
                            values=SCORE_FIELDS, aggfunc="first")
    piv.columns = [f"{field}__{judge}" for field, judge in piv.columns]
    df = gens.merge(piv.reset_index(), on="gen_id", how="inner")
    for field in SCORE_FIELDS:
        cols = [c for c in df.columns if c.startswith(field + "__")]
        df[field] = (df[cols].mean(axis=1) >= 0.5).astype(int)
    return df


def gwet_ac1_binary(r1, r2) -> float:
    """Gwet's AC1 for two binary raters (prevalence-robust alternative to kappa)."""
    r1, r2 = np.asarray(r1, float), np.asarray(r2, float)
    po = np.mean(r1 == r2)
    pi1 = (r1.mean() + r2.mean()) / 2.0
    pe = 2 * pi1 * (1 - pi1)
    return 1.0 if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)


def block_bootstrap_ci(df: pd.DataFrame, value_col: str, group_cols=None,
                       cluster_col: str = "claim_id", n_boot: int = 1000,
                       seed: int = 42, alpha: float = 0.05) -> pd.DataFrame:
    """95% CI for a proportion, resampling whole claims (preserves within-claim
    correlation across the 5 repeated runs)."""
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    group_cols = group_cols or []

    def point(frame):
        return frame.groupby(group_cols)[value_col].mean() if group_cols else pd.Series({"all": frame[value_col].mean()})

    boots = []
    for _ in range(n_boot):
        samp = rng.choice(clusters, size=len(clusters), replace=True)
        boots.append(point(df[df[cluster_col].isin(samp)]))
    boot_df = pd.concat(boots, axis=1)
    out = pd.DataFrame({
        "mean": point(df) * 100,
        "ci_lo": boot_df.quantile(alpha / 2, axis=1) * 100,
        "ci_hi": boot_df.quantile(1 - alpha / 2, axis=1) * 100,
    }).round(1)
    return out


def cohen_h(p1: float, p2: float) -> float:
    """Scale-invariant effect size for two proportions (0.2/0.5/0.8 = S/M/L)."""
    return abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def tost_equivalence_by_claim(df: pd.DataFrame, level_col: str = "controversy_level",
                              value_col: str = "refusal", margin: float = 0.05,
                              cluster_col: str = "claim_id", n_boot: int = 1000,
                              seed: int = 42, alpha: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    levels = sorted(df[level_col].unique())

    def level_means(frame):
        return frame.groupby(level_col)[value_col].mean()

    boot = [level_means(df[df[cluster_col].isin(
                rng.choice(clusters, size=len(clusters), replace=True))])
            for _ in range(n_boot)]
    boot = pd.concat(boot, axis=1).reindex(levels)

    rows = []
    for i, a in enumerate(levels):
        for b in levels[i + 1:]:
            diffs = (boot.loc[a] - boot.loc[b]).dropna().values
            lo, hi = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])  # 90% CI
            rows.append({
                "level_a": a, "level_b": b,
                "obs_diff_pp": round((df[df[level_col] == a][value_col].mean()
                                      - df[df[level_col] == b][value_col].mean()) * 100, 2),
                "ci90_lo_pp": round(lo * 100, 2), "ci90_hi_pp": round(hi * 100, 2),
                "margin_pp": margin * 100,
                "equivalent": bool(lo > -margin and hi < margin),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="DeflectBench reproducible stats")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    d = load_consensus(args.data_dir)
    print(f"Loaded {len(d):,} consensus-scored generations\n")
    print("Block-bootstrap refusal CIs by model:")
    print(block_bootstrap_ci(d, "refusal", group_cols=["model"]).to_string(), "\n")
    print("TOST: refusal equivalence across controversy levels (+-5pp):")
    print(tost_equivalence_by_claim(d).to_string(index=False))
