"""Export Gemma 4 hidden-rule Game of 24 results for the dashboard.

Run from experiments/game24_custom:
    python export_dashboard_data.py
Writes outputs/dashboard/curves.json (one row per run x round) and
outputs/dashboard/summary.json (one row per run), each as {rows, meta}.
"""
import glob
import json
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
MODELS = ["gemma4-26b", "qwen3.8-27b"]
ENVS = {"normal": "normal-on-ops_swap", "ops_swap": "ops_swap", "base3": "base3", "base7": "base7"}
VARIANTS = {  # variant -> experiment it belongs to
    "official": "baseline",
    "trust": "instruction suffix",
    "must_believe": "instruction suffix",
    "reward_only": "prompt prefix 2x2",
    "new_world": "prompt prefix 2x2",
    "both": "prompt prefix 2x2",
    "hint_base": "base hint",
    "hint_base7": "base hint",
}


def latest_run(prefix, model):
    runs = [r for r in sorted(glob.glob(str(HERE / "outputs" / f"{prefix}_api_{model}_*")))
            if Path(r, "summary.json").exists()]
    return Path(runs[-1]) if runs else None


def pct(x):
    return round(100 * x, 2)


def main():
    curves, summary = [], []
    for model, variant, env in [(m, v, e) for m in MODELS for v in VARIANTS for e in ENVS]:
        experiment = VARIANTS[variant] if model == MODELS[0] else f"{model} replication"
        base = ENVS[env]
        run = latest_run(base if variant == "official" else f"{base}_prompt-{variant}", model)
        if run is None:
            continue
        rounds = json.load(open(run / "summary.json"))
        outputs = json.load(open(run / "outputs.json"))
        for r in rounds:
            curves.append({"model": model, "variant": variant, "env": env, "round": r["round"], "mode": r["mode"],
                           "solve_pct": pct(r["solve_rate"]), "mean_reward": round(r["mean_reward"], 3)})
        exploit = [r["solve_rate"] for r in rounds if r["round"] % 2 == 1]
        resolved = []  # after a puzzle's first solve: solved again in later exploit rounds?
        for s in outputs:
            first = next((o["round"] for o in s["output"] if o["solved"]), None)
            if first is not None:
                resolved += [o["solved"] for o in s["output"] if o["round"] > first and o["round"] % 2 == 1]
        summary.append({
            "key": f"{model}|{variant}|{env}", "model": model, "experiment": experiment, "variant": variant, "env": env,
            "rounds_done": len(rounds),
            "exploit_first5_pct": pct(st.mean(exploit[:5])) if len(exploit) >= 5 else None,  # rounds 1-9
            "exploit_mean_pct": pct(st.mean(exploit)) if exploit else None,
            "exploit_last10_pct": pct(st.mean(exploit[-10:])) if exploit else None,
            "resolve_pct": pct(st.mean(resolved)) if resolved else None,
            "solved_once_pct": pct(st.mean([any(o["solved"] for o in s["output"]) for s in outputs])),
        })
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta = [
        {"label": "export_dashboard_data.py", "value": Path(__file__).read_text(), "type": "code", "lang": "python"},
        {"label": "Input", "value": f"outputs/<run>/summary.json and outputs.json of the {', '.join(MODELS)} runs "
                                    "(100 puzzles x 51 rounds per run)"},
        {"label": "Exported", "value": now},
    ]
    out = HERE / "outputs" / "dashboard"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": curves, "meta": meta}, open(out / "curves.json", "w"))
    json.dump({"rows": summary, "meta": meta}, open(out / "summary.json", "w"))
    print(f"{len(curves)} curve rows, {len(summary)} runs -> {out}  ({now})")


if __name__ == "__main__":
    main()
