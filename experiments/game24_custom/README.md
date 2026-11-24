# Hidden-rule Game of 24

Can an LLM discover changed rules of a familiar task **from reward alone**, through in-context RL?

The model gets the **original** Game of 24 ICRL prompt: the same few-shot file (`../game24/game24_3steps.txt`), the same explore/exploit instructions and the same `<attempt>` format as `../game24/game24_icrl.py`. It is never told that anything has changed. The environment secretly applies different rules and returns only the reward.

This directory is self-contained. It does not modify or import any other code in the repository; it only reads the few-shot file from `../game24/`.

## Environments

| `--env` | Hidden rule | What the model sees |
|---|---|---|
| `normal` | none (control) | normal Game of 24 on the `ops_swap` puzzle set (`--dataset ops_swap`) |
| `ops_swap` | `+`↔`-` and `*`↔`/` are swapped (`a + b` is really `a − b`) | normal-looking puzzles |
| `base3` | the puzzle numbers are written in base 3; the target is the true value 24 | normal-looking puzzles |
| `base7` | the puzzle numbers **and** the target "24" are written in base 7, so the true target is 18 | normal-looking puzzles |

The puzzle sets are in `data/<env>.json`: 100 puzzles each, built by `build_datasets.py` with seed 0.
- Each puzzle is solvable under the hidden rule.
- Each puzzle also looks solvable under normal decimal rules.
- No normal-rule solution also solves the puzzle under the hidden rule ("no leak").

### Reward

Rewards are computed exactly in code (`game24_envs.py`), using Fraction arithmetic and a brute-force solver. An LLM judge is not used because it would apply the normal rules.

- **Each of the 3 steps:** +3 if the **true** remaining numbers can still reach the target, otherwise 0.
- **After the third step:** +10 if the true final value equals the target.
- **Maximum:** 19.
- **Labels vs. true values:** the result the model writes for a step (e.g. `= 10`) is used only as a label for that number in later steps. The environment computes the true value itself.

## Prompt variants

Defined in `configs/prompt_variants.json`.
- `prompt_prefix` is put at the very start of the prompt.
- `extra_instruction` is appended after the explore/exploit instruction in every round.

| Experiment | Variant | Added text |
|---|---|---|
| baseline | `official` | (none — original prompt) |
| 1: instruction suffix | `trust` | "You can trust the reward." |
| 1: instruction suffix | `must_believe` | "You have to believe the reward." |
| 2: prompt prefix (2×2) | `reward_only` | "The reward is the only reliable signal." |
| 2: prompt prefix (2×2) | `new_world` | "The rules of this world may be different from what you are used to." |
| 2: prompt prefix (2×2) | `both` | `new_world` + `reward_only` |
| 4: base hint | `hint_base` | "The numbers in this world are not written in base 10. Keep writing every number, including intermediate results, in the same notation as the input." |
| 4: base hint (upper bound) | `hint_base7` | "All numbers, including the target 24, are written in base 7. Keep writing every number, including intermediate results, in base 7." |

Experiment 3 reruns base7 `official` vs `both` with a second model family.

## Running

```bash
cd experiments/game24_custom

# Pipeline check (no GPU or API needed)
python game24_hidden_icrl.py --env base7 --backend mock --rounds 3 --num_samples 5

# Against an OpenAI-compatible server (e.g. `vllm serve` or llama.cpp `llama-server` on :8000)
OPENAI_API_KEY=EMPTY python game24_hidden_icrl.py --env base7 --prompt_variant both \
    --backend api --base_url http://localhost:8000/v1 --model <served-model-name> --think --concurrency 16

# Control: normal rules on the ops_swap puzzles
python game24_hidden_icrl.py --env normal --dataset ops_swap ...

# Rebuild the puzzle sets (optional; the committed data/ is what the results use)
python build_datasets.py
```

**Defaults** match the original setup: 100 puzzles × 51 rounds, temperature 0.6, top_p 0.95, max_tokens 1000.
- Even rounds explore; odd rounds exploit.
- Each puzzle keeps its own context with all of its past attempts and rewards.
- `--think` drops the original `/no_think` suffix.

**Outputs** go to `outputs/<env>[_prompt-<variant>]_<backend>_<model>_<timestamp>/`:
- `config.json`
- `summary.json`: per-round mean reward and solve rate
- `outputs.json`: every prompt, response and step reward, plus a `debug_log` with the true values (for analysis only; it is never shown to the model)

**Analysis scripts** (need `matplotlib`, which is not in `requirements/`):

| Script | Output |
|---|---|
| `plot_exploit.py` | per-round exploit success rate per env × variant |
| `plot_reward_band.py` | per-round mean true reward with a 95% CI band over the 100 puzzles |
| `export_dashboard_data.py` | `outputs/dashboard/{curves,summary}.json`: one row per run × round, and one row per run |

Only `outputs/plots/` and `outputs/dashboard/` are committed. The raw run directories are about 53 MB each, so they are git-ignored.

## Results

**Models**
- Gemma 4 26B (mixture-of-experts): W4A16, served with vLLM.
- Qwen3.8-27B: Q5_K_M GGUF, served with llama.cpp with thinking disabled.

**Metrics** (all exploit metrics use odd rounds only)
- **exploit first5:** success rate over exploit rounds 1–9.
- **exploit mean:** success rate over all 25 exploit rounds.
- **exploit last10:** success rate over exploit rounds 31–49.
- **re-solve:** success rate in later exploit rounds, after a puzzle's first solve.
- **solved once:** share of puzzles solved at least once.

### Gemma 4 26B, all environments (exploit mean % / re-solve %)

| Variant | normal | ops_swap | base3 | base7 |
|---|---|---|---|---|
| official | 61.2 / 82.7 | 0.8 / 6.2 | 0.0 / 0.0 | 3.2 / 7.7 |
| trust | 68.0 / 89.3 | 0.6 / 6.4 | 0.0 / 0.0 | 3.4 / 8.4 |
| must_believe | 62.4 / 85.0 | 0.2 / 1.9 | 0.1 / 9.4 | 3.5 / 7.6 |
| reward_only | 66.6 / 87.8 | 1.0 / 12.0 | 0.1 / 8.3 | 6.8 / 18.1 |
| new_world | 57.4 / 81.3 | 0.5 / 5.0 | 0.0 / 0.0 | 4.5 / 12.5 |
| both | 61.2 / 85.4 | 0.7 / 6.0 | 0.3 / 15.8 | 9.1 / 27.7 |

### base7 in detail (exploit first5 / mean / last10 %, re-solve %, solved once %)

| Model | Variant | first5 | mean | last10 | re-solve | solved once |
|---|---|---|---|---|---|---|
| Gemma | official | 5.2 | 3.2 | 2.6 | 7.7 | 44 |
| Gemma | both | 7.8 | 9.1 | 9.0 | 27.7 | 40 |
| Gemma | hint_base | 4.6 | 4.3 | 4.3 | 10.3 | 54 |
| Gemma | hint_base7 | 14.0 | 25.7 | 31.5 | 58.2 | 60 |
| Qwen | official | 6.2 | 3.8 | 3.0 | 10.1 | 49 |
| Qwen | both | 10.8 | 10.6 | 10.2 | 24.4 | 55 |

### Findings

1. **The hidden rules are not discovered from reward alone.**
   - Under `ops_swap` and `base3`, success stays near 0% for all 51 rounds.
   - Under `base7` with the original prompt, exploit success drifts down. The share of answers that claim "= 24" rises over rounds, from about 30% to about 75%: the model moves back toward its prior.
2. **Prompt framing changes exploitation, not rule discovery.**
   - On base7, `both` roughly triples exploit success. Gemma goes from 3.2% to 9.1%; Qwen goes from 3.8% to 10.6%, so the effect replicates across model families.
   - The curves stay flat from round 1. The gain comes from re-solving: the model keeps repeating an answer that once got 19, even when that answer looks wrong to it.
3. **The model notices the anomaly but explains it away.** In its reasoning it writes, for example: "Reward 19 but the result is 18. This is weird… maybe a typo." It then returns to targeting 24.
4. **Saying "not base 10" is not enough.**
   - Under `hint_base`, not a single step in 51 rounds uses base-7 arithmetic; the model never forms a "which base?" hypothesis.
   - Under `hint_base7`, the rule is given outright. Success rises to 31.5% in late rounds. However, only about 16% of steps use correct base-7 arithmetic, and this share does not improve over 51 rounds. The rise comes from keeping lucky solves (re-solve 58%), not from better base-7 computation.
   - So base7 mixes two difficulties: forming the hypothesis (exploration) and computing in base 7 (execution).
