"""
plot_training.py
================
Generates reward-curve and loss-curve PNGs from logs/training_log.csv
produced by the _CSVLogCallback in train_rl.py.

Usage (after training finishes):
    python plot_training.py                        # default: logs/training_log.csv
    python plot_training.py --log logs/my_run.csv  # custom path
    python plot_training.py --out results/         # save PNGs to a different folder

Outputs
-------
    plots/reward_curve.png   — mean reward across SFT + GRPO steps
    plots/loss_curve.png     — training loss (SFT phase)
    plots/kl_curve.png       — KL divergence (GRPO phase)
    plots/combined.png       — all three side-by-side (for README embedding)
"""

import argparse
import os
import sys

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")          # headless — safe on HF Spaces / CI
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib pandas")
    sys.exit(1)


# ── Styling ───────────────────────────────────────────────────────────────────
COLORS = {
    "sft":  "#5b8dd9",   # blue
    "grpo": "#e06c75",   # red
    "reward": "#98c379", # green
    "kl":   "#d19a66",   # orange
}
FONT = {"family": "DejaVu Sans", "size": 11}


def load(log_path: str) -> pd.DataFrame:
    if not os.path.exists(log_path):
        print(f"ERROR: {log_path} not found. Run training first.")
        sys.exit(1)
    df = pd.read_csv(log_path)
    # Coerce numeric columns (they may be empty strings)
    for col in ("loss", "reward_mean", "reward_std", "kl"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"Loaded {len(df)} rows from {log_path}")
    print(df.groupby("phase")[["reward_mean", "loss"]].describe().round(4))
    return df


def plot_reward(df: pd.DataFrame, out_dir: str):
    """Mean reward across SFT + GRPO steps on one chart."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for phase, color in [("sft", COLORS["sft"]), ("grpo", COLORS["grpo"])]:
        sub = df[df["phase"] == phase].dropna(subset=["reward_mean"])
        if sub.empty:
            continue
        ax.plot(sub["step"], sub["reward_mean"], color=color, label=f"{phase.upper()} reward", lw=1.5)
        # Rolling average smoothing
        if len(sub) >= 5:
            smooth = sub["reward_mean"].rolling(5, min_periods=1).mean()
            ax.plot(sub["step"], smooth, color=color, lw=2.5, linestyle="--",
                    label=f"{phase.upper()} reward (smoothed)")
        if "reward_std" in sub.columns:
            std = sub["reward_std"].fillna(0)
            ax.fill_between(sub["step"],
                            sub["reward_mean"] - std,
                            sub["reward_mean"] + std,
                            alpha=0.15, color=color)
    ax.axhline(0, color="grey", lw=0.8, linestyle=":")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Reward Curve — ModelCardAudit RL Training")
    ax.legend(fontsize=9)
    plt.rc("font", **FONT)
    plt.tight_layout()
    path = os.path.join(out_dir, "reward_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_loss(df: pd.DataFrame, out_dir: str):
    """SFT training loss."""
    sft = df[(df["phase"] == "sft")].dropna(subset=["loss"])
    if sft.empty:
        print("No SFT loss data found — skipping loss curve.")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sft["step"], sft["loss"], color=COLORS["sft"], lw=1.5, label="SFT loss")
    smooth = sft["loss"].rolling(5, min_periods=1).mean()
    ax.plot(sft["step"], smooth, color=COLORS["sft"], lw=2.5, linestyle="--", label="Smoothed")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("SFT Warmup — Training Loss")
    ax.legend(fontsize=9)
    plt.rc("font", **FONT)
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_kl(df: pd.DataFrame, out_dir: str):
    """GRPO KL divergence."""
    grpo = df[(df["phase"] == "grpo")].dropna(subset=["kl"])
    if grpo.empty:
        print("No KL data found — skipping KL curve.")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grpo["step"], grpo["kl"], color=COLORS["kl"], lw=1.5, label="KL divergence")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("KL (nats)")
    ax.set_title("GRPO — KL Divergence (Policy vs Reference)")
    ax.legend(fontsize=9)
    plt.rc("font", **FONT)
    plt.tight_layout()
    path = os.path.join(out_dir, "kl_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_combined(df: pd.DataFrame, out_dir: str):
    """Single wide PNG with reward + loss + KL side-by-side for README."""
    fig = plt.figure(figsize=(16, 4))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    # --- Reward ---
    ax0 = fig.add_subplot(gs[0])
    for phase, color in [("sft", COLORS["sft"]), ("grpo", COLORS["grpo"])]:
        sub = df[df["phase"] == phase].dropna(subset=["reward_mean"])
        if sub.empty:
            continue
        ax0.plot(sub["step"], sub["reward_mean"], color=color, label=phase.upper(), lw=1.5)
        if len(sub) >= 5:
            ax0.plot(sub["step"], sub["reward_mean"].rolling(5, min_periods=1).mean(),
                     color=color, lw=2.5, linestyle="--")
    ax0.axhline(0, color="grey", lw=0.8, linestyle=":")
    ax0.set_title("Mean Reward"); ax0.set_xlabel("Step"); ax0.set_ylabel("Reward")
    ax0.legend(fontsize=8)

    # --- Loss ---
    ax1 = fig.add_subplot(gs[1])
    sft = df[df["phase"] == "sft"].dropna(subset=["loss"])
    if not sft.empty:
        ax1.plot(sft["step"], sft["loss"], color=COLORS["sft"], lw=1.5)
        ax1.plot(sft["step"], sft["loss"].rolling(5, min_periods=1).mean(),
                 color=COLORS["sft"], lw=2.5, linestyle="--")
    ax1.set_title("SFT Loss"); ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")

    # --- KL ---
    ax2 = fig.add_subplot(gs[2])
    grpo = df[df["phase"] == "grpo"].dropna(subset=["kl"])
    if not grpo.empty:
        ax2.plot(grpo["step"], grpo["kl"], color=COLORS["kl"], lw=1.5)
    ax2.set_title("KL Divergence (GRPO)"); ax2.set_xlabel("Step"); ax2.set_ylabel("KL (nats)")

    plt.suptitle("ModelCardAudit-Env — RL Training Metrics", fontsize=13, y=1.02)
    plt.rc("font", **FONT)
    plt.tight_layout()
    path = os.path.join(out_dir, "combined.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def compare_baseline(baseline_scores: dict, rl_scores: dict, out_dir: str):
    """
    Bar chart comparing baseline vs RL agent scores per task.
    Call this manually after running:
        python inference.py  (once without RL model → baseline)
        python inference.py  (once with RL model → rl)

    Example:
        from plot_training import compare_baseline
        compare_baseline(
            {"basic_completeness": 0.65, "technical_consistency": 0.40, "regulatory_compliance": 0.20},
            {"basic_completeness": 0.72, "technical_consistency": 0.55, "regulatory_compliance": 0.31},
            "plots/"
        )
    """
    tasks = list(baseline_scores.keys())
    base_vals = [baseline_scores[t] for t in tasks]
    rl_vals   = [rl_scores[t]       for t in tasks]

    x = range(len(tasks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_b = ax.bar([i - width/2 for i in x], base_vals, width, label="Deterministic Baseline",
                    color=COLORS["sft"], alpha=0.85)
    bars_r = ax.bar([i + width/2 for i in x], rl_vals,   width, label="RL Agent (GRPO)",
                    color=COLORS["grpo"], alpha=0.85)

    def _label(bars):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
    _label(bars_b); _label(bars_r)

    ax.set_xticks(list(x))
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Final Score")
    ax.set_title("Baseline vs RL Agent — Score Comparison per Task")
    ax.legend()
    plt.rc("font", **FONT)
    plt.tight_layout()
    path = os.path.join(out_dir, "baseline_vs_rl.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from training_log.csv")
    parser.add_argument("--log", default="logs/training_log.csv", help="Path to training_log.csv")
    parser.add_argument("--out", default="plots/", help="Output directory for PNG files")
    # Optional: pass baseline and RL scores to generate comparison bar chart
    parser.add_argument("--baseline", type=str, default=None,
                        help='JSON string of baseline scores e.g. \'{"basic_completeness":0.65,...}\'')
    parser.add_argument("--rl",       type=str, default=None,
                        help='JSON string of RL scores e.g. \'{"basic_completeness":0.72,...}\'')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load(args.log)

    plot_reward(df, args.out)
    plot_loss(df, args.out)
    plot_kl(df, args.out)
    plot_combined(df, args.out)

    if args.baseline and args.rl:
        compare_baseline(
            json.loads(args.baseline),
            json.loads(args.rl),
            args.out,
        )

    print(f"\nAll plots saved to: {os.path.abspath(args.out)}")
    print("Embed plots/combined.png and plots/baseline_vs_rl.png in your README.")


if __name__ == "__main__":
    import json
    main()
