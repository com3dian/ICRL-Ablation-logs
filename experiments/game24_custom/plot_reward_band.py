"""Per-round mean true reward with a 95% CI band, one figure per environment.

    python plot_reward_band.py                                   # Experiment 2: base7, ops_swap, normal
    python plot_reward_band.py --envs base7 --all_rounds         # include explore rounds too
    python plot_reward_band.py --variants official trust must_believe --tag exp1

True reward = the code-computed reward of each answer (0-19: 3 per step that keeps
the target reachable, +10 if solved). Band = mean +/- 1.96 standard errors over the
100 puzzles. By default only exploit rounds (odd rounds) are plotted.
"""
import argparse
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ENVS = {"base7": ("base7", "base7 (base-7 numbers, true target 18)"),
        "ops_swap": ("ops_swap", "ops_swap (+ - and * / flipped)"),
        "normal": ("normal-on-ops_swap", "normal rules (control)")}
COLORS = {"official": "#0072b2", "reward_only": "#d55e00", "new_world": "#009e73", "both": "#cc79a7",
          "trust": "#e69f00", "must_believe": "#56b4e9", "hint_base": "#e69f00", "hint_base7": "#009e73"}


def load_outputs(prefix, model):
    runs = [r for r in sorted(glob.glob(str(HERE / "outputs" / f"{prefix}_api_{model}_*")))
            if Path(r, "outputs.json").exists()]
    return json.load(open(Path(runs[-1], "outputs.json"))) if runs else None


def per_round(outputs, all_rounds):
    by_round = {}
    for s in outputs:
        for o in s["output"]:
            if all_rounds or o["round"] % 2 == 1:
                by_round.setdefault(o["round"], []).append(o["total"])
    xs = sorted(by_round)
    means = [st.mean(by_round[r]) for r in xs]
    half = [1.96 * st.stdev(by_round[r]) / len(by_round[r]) ** 0.5 for r in xs]
    return xs, means, half


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", nargs="+", default=["base7", "ops_swap", "normal"], choices=list(ENVS))
    p.add_argument("--variants", nargs="+", default=["official", "reward_only", "new_world", "both"])
    p.add_argument("--model", default="gemma4-26b")
    p.add_argument("--all_rounds", action="store_true", help="also plot explore (even) rounds")
    p.add_argument("--tag", default="exp2")
    args = p.parse_args()

    out_dir = HERE / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for env in args.envs:
        prefix, title = ENVS[env]
        fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
        for v in args.variants:
            outputs = load_outputs(prefix if v == "official" else f"{prefix}_prompt-{v}", args.model)
            if outputs is None:
                print(f"skip {v} {env}: no run found")
                continue
            xs, m, h = per_round(outputs, args.all_rounds)
            c = COLORS.get(v, None)
            ax.fill_between(xs, [a - b for a, b in zip(m, h)], [a + b for a, b in zip(m, h)],
                            color=c, alpha=0.15, linewidth=0)
            ax.plot(xs, m, color=c, lw=2, marker="o", ms=4, label=v)
        rounds = "all rounds" if args.all_rounds else "exploit rounds only"
        ax.set_title(f"Game of 24 · {title} · {args.model} · prompt prefix", fontsize=12)
        ax.set_xlabel(f"Round ({rounds})")
        ax.set_ylabel("Mean true reward (0–19)")
        ax.set_xlim(0, 51 if args.all_rounds else 50)
        ax.set_ylim(bottom=0)
        ax.grid(color="#e6e6e6", lw=0.8)
        ax.legend(title="Prompt prefix", loc="lower right" if env == "normal" else "upper right", frameon=False)
        fig.text(0.99, 0.01, "band: 95% CI over 100 puzzles", ha="right", va="bottom", fontsize=8, color="#666")
        fig.tight_layout()
        path = out_dir / f"reward_band_{args.tag}_{env}{'_all' if args.all_rounds else ''}.png"
        fig.savefig(path)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
