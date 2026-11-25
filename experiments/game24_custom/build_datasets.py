"""Build the fixed puzzle sets for the hidden-rule Game of 24 environments.

    cd experiments/game24_custom
    python build_datasets.py            # 100 puzzles per env, seed 0
    python build_datasets.py --n 200 --seed 1

Writes data/<env_name>.json. Only "question" is shown to the model; the other
fields are for analysis.
"""
import argparse
import json

from game24_envs import DATA_DIR, HIDDEN_RULE_ENVS, fmt, make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_value", type=int, default=13)
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    for name in HIDDEN_RULE_ENVS:  # the "normal" control reuses these puzzle sets
        env = make_env(name)
        puzzles, stats = env.build_dataset(n=args.n, max_value=args.max_value, seed=args.seed)
        out = {
            "env": name,
            "description": env.description,
            "base": env.base,
            "shown_target": "24",
            "true_target": fmt(env.target),
            "seed": args.seed,
            "max_value": args.max_value,
            "stats": stats,
            "puzzles": puzzles,
        }
        with open(DATA_DIR / f"{name}.json", "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"{name:9s} kept {stats['kept']:4d} / examined {stats['examined']:4d}  "
              f"(unsolvable {stats['unsolvable']}, decimal-unsolvable {stats['decimal_unsolvable']}, "
              f"leaky {stats['leaky']})  -> data/{name}.json")


if __name__ == "__main__":
    main()
