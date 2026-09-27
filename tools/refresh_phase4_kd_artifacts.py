#!/usr/bin/env python3
"""Sync Phase 4 KD metrics from the GPU and refresh paper figures + 12b notebook.

Run from repo root whenever distill training progresses or finishes:

  python tools/refresh_phase4_kd_artifacts.py

Requires: SSH to GPU_HOST, matplotlib, pandas, local `notebooks/12b_*.ipynb` template cells.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

GPU_HOST = "107.170.17.174"
GPU_USER = "root"
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
REMOTE_BONSAI = "/root/bonsai"

RUNS = {
    "vs_baseline": "20260927T171637Z-3ddb90-distill-aggressive-vs-baseline",
    "vs_corrected": "20260927T175820Z-2e6846-distill-aggressive-vs-corrected",
}

CACHE = ROOT / "runs/_notebook_cache/phase4"
FIG = ROOT / "docs/latex/figures/phase4_kd"
NOTEBOOK = ROOT / "notebooks/12b_phase4_distillation_training_curves.ipynb"


def ssh_cat(remote_path: str) -> str:
    argv = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-i",
        str(SSH_KEY),
        f"{GPU_USER}@{GPU_HOST}",
        f"cat {remote_path}",
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return proc.stdout


def sync_run(label: str, run_id: str) -> tuple[Path, Path]:
    CACHE.mkdir(parents=True, exist_ok=True)
    remote = f"{REMOTE_BONSAI}/runs/{run_id}"
    metrics = CACHE / f"{label}-metrics.jsonl"
    config = CACHE / f"{label}-config.json"
    metrics.write_text(ssh_cat(f"{remote}/metrics.jsonl"))
    config.write_text(ssh_cat(f"{remote}/config.json"))
    return metrics, config


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    FIG.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict] = {}

    for label, run_id in RUNS.items():
        m_path, c_path = sync_run(label, run_id)
        df = pd.read_json(m_path, lines=True)
        frames[label] = df
        cfg = json.loads(c_path.read_text())
        final = df.iloc[-1]
        val = df.dropna(subset=["val_loss"])
        summaries[label] = {
            "run_id": run_id,
            "steps_logged": len(df),
            "last_step": int(final["step"]),
            "complete": int(final["step"]) >= cfg.get("total_steps", 500) - 1,
            "final_train_loss": float(final["train_loss"]),
            "final_val_loss": float(val.iloc[-1]["val_loss"]) if len(val) else None,
            **{
                k: cfg[k]
                for k in ("peak_lr", "total_steps", "batch_size", "warmup_steps")
                if k in cfg
            },
        }

    # Per-arm loss curves
    colors = {"vs_baseline": "#2563eb", "vs_corrected": "#9333ea"}
    labels = {"vs_baseline": "uncorrected teacher", "vs_corrected": "LoRA-corrected teacher"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, df in frames.items():
        ax.plot(df["step"], df["train_loss"], label=f"{labels[key]} (train)", color=colors[key])
        val = df.dropna(subset=["val_loss"])
        if len(val):
            ax.plot(
                val["step"],
                val["val_loss"],
                ls="--",
                marker="o",
                ms=2.5,
                label=f"{labels[key]} (val)",
                color=colors[key],
            )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("forward KL (top-20 support)")
    ax.set_title("Phase 4 teacher ablation — KD training (matched 6000-sample caches)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(
            FIG / f"distill_teacher_ablation_comparison.{ext}", dpi=160, bbox_inches="tight"
        )
    plt.close(fig)

    # Final val_loss bar (where available)
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    names, vals = [], []
    for key in ("vs_baseline", "vs_corrected"):
        v = summaries[key].get("final_val_loss")
        if v is not None and summaries[key]["complete"]:
            names.append(labels[key])
            vals.append(v)
    if names:
        ax2.bar(names, vals, color=[colors["vs_baseline"], colors["vs_corrected"]][: len(names)])
        ax2.set_ylabel("final val KD loss")
        ax2.set_title("Completed runs — validation KD")
        fig2.tight_layout()
        for ext in ("pdf", "png"):
            fig2.savefig(FIG / f"distill_final_val_kd_bar.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig2)

    results = {
        "updated_utc": subprocess.check_output(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True
        ).strip(),
        "runs": summaries,
    }
    (FIG / "results.json").write_text(json.dumps(results, indent=2))

    notes = f"""# Phase 4 KD figures — notes for paper integration

Auto-updated by `tools/refresh_phase4_kd_artifacts.py` ({results["updated_utc"]}).

## Experimental setup (both arms)

- **Student:** same pruned aggressive depth+experts candidate (~8.5B params).
- **Teachers:** cached top-20 logits on **identical** 6000 TypeScript samples (seed 11).
  - Baseline: `Qwen/Qwen3-30B-A3B-Base`
  - Corrected: `sartajbhuvaji/bonsai-qwen3-30b-a3b-teacher-corrected-lora`
- **Loss:** forward KL on teacher top-k support (not full-vocab CE).
- **Training:** AdamW 8-bit, 500 steps, batch 4, peak_lr=2e-4, warmup 75 (see `config.json` per run).

## Results snapshot

```json
{json.dumps(summaries, indent=2)}
```

## Interpretation (loss-only; eval still needed)

- Lower **val KD** on the matched holdout means the student matched that teacher's distribution better.
- This does **not** by itself prove HumanEval/MBPP ranking — add eval bars when available.

## Figures in this directory

| File | Use |
|------|-----|
| `distill_teacher_ablation_comparison.pdf` | Main training-curve panel (both teachers) |
| `distill_vs_baseline_kd_loss.pdf` | Baseline arm only (legacy single-panel) |
| `distill_final_val_kd_bar.pdf` | Bar chart once **both** runs hit 500 steps |
| `results.json` | Machine-readable final metrics for agents/LaTeX tables |

## Paper agent TODO

- [ ] Re-run refresh script after 3c completes; include bar chart with both teachers.
- [ ] Optional: small HumanEval-TS / MBPP-TS sample on both distilled checkpoints.
- [ ] Caption: top-k truncated KD; same student init; matched caches.
"""
    (FIG / "PAPER_NOTES.md").write_text(notes)

    # Embed plot in notebook (update RUNS + committed snapshot outputs)
    df_b = frames["vs_baseline"]
    console_tail = "metrics.jsonl tail (baseline arm):\n" + df_b.tail(6).to_string(index=False)
    if "vs_corrected" in frames:
        console_tail += "\n\nmetrics.jsonl tail (corrected arm):\n" + frames["vs_corrected"].tail(
            6
        ).to_string(index=False)

    buf = BytesIO()
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, df in frames.items():
        ax.plot(df["step"], df["train_loss"], label=labels[key], color=colors[key])
    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("KD train loss")
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    plot_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    compare_msg = "Generated comparison figures under docs/latex/figures/phase4_kd/\n" + json.dumps(
        summaries, indent=2
    )

    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["id"] == "cfg":
            src = "".join(cell["source"])
            cell["source"] = [
                ln + ("\n" if not ln.endswith("\n") and i < len(src.splitlines()) - 1 else "")
                for i, ln in enumerate(
                    src.replace(
                        '"vs_corrected": None',
                        f'"vs_corrected": "{RUNS["vs_corrected"]}"',
                    ).splitlines()
                )
            ]
            if cell["source"] and not cell["source"][-1].endswith("\n"):
                pass
            # fix source as list of lines with newlines per nbformat
            new_src = []
            for line in src.replace(
                '"vs_corrected": None', f'"vs_corrected": "{RUNS["vs_corrected"]}"'
            ).split("\n"):
                new_src.append(line + "\n")
            if new_src:
                new_src[-1] = new_src[-1].rstrip("\n")
            cell["source"] = new_src
        if cell["id"] == "sync" and cell["cell_type"] == "code":
            cell["outputs"] = [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [ln + "\n" for ln in compare_msg.split("\n")],
                }
            ]
        if cell["id"] == "plot-baseline":
            cell["outputs"] = [
                {"data": {"image/png": plot_b64}, "metadata": {}, "output_type": "display_data"},
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [f"wrote {FIG / 'distill_teacher_ablation_comparison.pdf'}\n"],
                },
            ]
        if cell["id"] == "plot-compare":
            cell["outputs"] = [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Comparison plot refreshed via tools/refresh_phase4_kd_artifacts.py\n"
                    ],
                }
            ]

    NOTEBOOK.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Updated {FIG} and {NOTEBOOK}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
