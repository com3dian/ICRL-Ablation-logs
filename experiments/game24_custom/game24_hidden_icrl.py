"""ICRL on hidden-rule Game of 24 (ops_swap / base3 / base7).

The prompt is built exactly like experiments/game24/game24_icrl.py (same few-shot
file, same instructions, same <attempt> format with per-step rewards). Only two
things differ:
    1. puzzles come from data/<env>.json (built by build_datasets.py)
    2. rewards come from game24_envs.evaluate() instead of an LLM judge

Examples:
    # pipeline check, no GPU / API needed
    python game24_hidden_icrl.py --env base7 --backend mock --rounds 3 --num_samples 5

    # local vLLM (raw-text prompt, like the original)
    python game24_hidden_icrl.py --env ops_swap --backend vllm --model Qwen/Qwen3-8B

    # any OpenAI-compatible API (OpenRouter, a `vllm serve` server, OpenAI)
    OPENROUTER_API_KEY=... python game24_hidden_icrl.py --env base3 --backend api \
        --base_url https://openrouter.ai/api/v1 --api_key_env OPENROUTER_API_KEY \
        --model qwen/qwen3-32b
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from game24_envs import ENVS, HIDDEN_RULE_ENVS, _normal_plan, load_dataset, make_env

HERE = Path(__file__).parent
FEW_SHOT_FILE = HERE.parent / "game24" / "game24_3steps.txt"

# ---- instructions: copied verbatim from experiments/game24/game24_icrl.py ----
exploitation_instruction = "Instruction: You will be given multiple <attempt>…</attempt> examples, each showing a candidate Response, and the Rewards for each step of the Response. Your task: Based on the previous attempts, try your best to produce a response that can achieve higher rewards, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format."

exploration_instruction = "Instruction: Examine all the `<attempt>…</attempt>` examples, each showing a candidate Response, the Reward for each step of the Response, and a total Return of the steps.  Provide a response that is completely different for any steps from every single one of the previous attempts demonstrated in the context, while making sure it correctly follows the task instruction. Make sure the response correctly follows the task instruction, and put it in `<answer>**Response** Step1: ... Step2: ... Step3: ... **Answer**: <math operations of the 4 input numbers = 24></answer>` format."

format_instruction = " Only make one attempt, and put your answer in `<answer>**Response** Step1: ... (left: ...) Step2: ... (left: ...) Step3: ... (left: ...) **Answer**: <math operations of the 4 input numbers, even if it does not equal 24></answer>` format. Whether the Answer is correct or incorrect, do not try again. \n"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=list(ENVS), required=True)
    p.add_argument("--dataset", choices=HIDDEN_RULE_ENVS, default=None,
                   help="puzzle set to use (default: same as --env; required for --env normal)")
    p.add_argument("--rounds", type=int, default=51)              # original: n=51
    p.add_argument("--num_samples", type=int, default=100)        # original: 100 puzzles
    p.add_argument("--max_attempts_in_context", type=int, default=None,
                   help="keep only the last K attempts in context (original: all)")
    p.add_argument("--no_reward", action="store_true", help="ablation: show attempts without rewards")
    p.add_argument("--zero_reward", action="store_true", help="ablation: show every total reward as 0")
    p.add_argument("--think", action="store_true", help="drop the /no_think suffix (original uses /no_think)")
    p.add_argument("--prompt_config", default=str(HERE / "configs" / "prompt_variants.json"),
                   help="JSON file defining prompt variants")
    p.add_argument("--prompt_variant", default="official",
                   help="variant name in --prompt_config: its prompt_prefix goes at the start of the prompt, "
                        "its extra_instruction after the explore/exploit instruction ('official' = original prompt)")
    # generation
    p.add_argument("--backend", choices=["vllm", "api", "mock"], default="vllm")
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--temperature", type=float, default=0.6)      # original sampling params
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=1000)
    # vllm backend
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--kv_cache_dtype", default="auto", help='e.g. "fp8" to fit longer contexts on 24GB')
    # api backend
    p.add_argument("--base_url", default=None)
    p.add_argument("--api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--api_mode", choices=["chat", "completion"], default="chat",
                   help="chat: prompt as one user message; completion: raw text like the original vLLM path")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--enable_thinking", action="store_true",
                   help="api/chat only: pass chat_template_kwargs enable_thinking=True. Start the server "
                        "with a reasoning parser (e.g. --reasoning-parser gemma4) so thinking is kept out "
                        "of the answer; raise --max_tokens too")
    # output
    p.add_argument("--output_dir", default=str(HERE / "outputs"))
    p.add_argument("--tag", default="")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Generation backends
# ---------------------------------------------------------------------------
def make_generator(args, env):
    """Returns gen(prompts) -> list of (text, reasoning). reasoning is '' unless thinking is enabled."""
    if args.backend == "mock":
        # Pretends to be a model that believes normal arithmetic: always writes a
        # correct-looking normal solution. Only for checking the pipeline.
        def gen(prompts):
            outs = []
            for pr in prompts:
                q = pr.rsplit("**Prompt**: Input: ", 1)[1].strip()
                plan = _normal_plan(q) or ["1 + 1 = 2", "2 + 2 = 4", "4 + 4 = 8"]
                body = " ".join(f"Step{k + 1}: {s} (left: ...)" for k, s in enumerate(plan))
                outs.append((f"<answer>**Response** {body} **Answer**: ... = 24</answer>", ""))
            return outs
        return gen

    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        kw = dict(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
                  gpu_memory_utilization=args.gpu_memory_utilization, kv_cache_dtype=args.kv_cache_dtype)
        if args.max_model_len:
            kw["max_model_len"] = args.max_model_len
        llm = LLM(**kw)
        sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)

        def gen(prompts):
            return [(o.outputs[0].text, "") for o in llm.generate(prompts, sp)]
        return gen

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv(args.api_key_env), base_url=args.base_url)

    def one(prompt):
        for attempt in range(6):
            try:
                if args.api_mode == "chat":
                    extra = {"chat_template_kwargs": {"enable_thinking": True}} if args.enable_thinking else None
                    r = client.chat.completions.create(
                        model=args.model, messages=[{"role": "user", "content": prompt}],
                        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                        extra_body=extra)
                    msg = r.choices[0].message
                    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
                    return msg.content or "", reasoning
                r = client.completions.create(
                    model=args.model, prompt=prompt,
                    temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
                return r.choices[0].text or "", ""
            except Exception as e:  # rate limits, timeouts
                wait = 2 ** attempt
                print(f"API error ({e}); retry in {wait}s")
                time.sleep(wait)
        return "", ""

    def gen(prompts):
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            return list(pool.map(one, prompts))
    return gen


# ---------------------------------------------------------------------------
# Prompt (same structure as experiments/game24/game24_icrl.py)
# ---------------------------------------------------------------------------
def format_attempt(demo, args):
    s = "<attempt>\n"
    s += f"**Prompt**: Input: {demo['prompt']}\n"
    ans, r = demo["answer"], demo["step_reward_strs"]
    if not args.no_reward:
        try:
            body = "**Response** Step" + ans.split("Step")[1]
            body += f"<Reward: {r[0]}> Step" + ans.split("Step")[2]
            body += f"<Reward: {r[1]}> Step" + ans.split("Step")[3].split("**Answer**")[0]
            body += f"<Reward: {r[2]}> **Answer**" + ans.split("**Answer**")[1]
            s += body
        except IndexError:  # malformed answer: show it without per-step rewards
            s += ans + "\n"
    else:
        s += ans + "\n"
    if not args.no_reward:
        s += f"**Reward**: {0.0}\n" if args.zero_reward else f"**Reward**: {demo['return_str']}. \n"
    s += "</attempt>"
    return s


def build_prompt(few_shot, sample, round_idx, args):
    prompt = (args.prompt_prefix + "\n\n" if args.prompt_prefix else "") + few_shot
    demos = sample["weak_demos"]
    if args.max_attempts_in_context is not None:
        demos = demos[-args.max_attempts_in_context:] if args.max_attempts_in_context > 0 else []
    for demo in demos:
        prompt += format_attempt(demo, args)
    prompt += "<instructions>\n"
    prompt += exploitation_instruction if round_idx % 2 == 1 else exploration_instruction
    if args.extra_instruction:
        prompt += " " + args.extra_instruction
    prompt += format_instruction
    if not args.think:
        prompt += "/no_think"
    prompt += "\n</instructions>\n"
    prompt += "<attempt>\n"
    prompt += f"**Prompt**: Input: {sample['question']}\n"
    return prompt


def extract_answer(generated):
    """Same as the original: text inside the first <answer>...</answer>, else ''."""
    m = re.findall(r"<answer>(.*?)</answer>", generated, re.DOTALL)
    return m[0] if m else ""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    env = make_env(args.env)
    if args.dataset is None:
        assert args.env != "normal", "--env normal needs --dataset (whose puzzles to play normally)"
        args.dataset = args.env
    questions = load_dataset(args.dataset)[: args.num_samples]
    with open(args.prompt_config) as f:
        variants = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    if args.prompt_variant not in variants:
        raise SystemExit(f"unknown --prompt_variant {args.prompt_variant!r}; choices: {list(variants)}")
    args.extra_instruction = variants[args.prompt_variant].get("extra_instruction", "")
    args.prompt_prefix = variants[args.prompt_variant].get("prompt_prefix", "")
    few_shot = FEW_SHOT_FILE.read_text()
    generate = make_generator(args, env)

    env_tag = args.env if args.dataset == args.env else f"{args.env}-on-{args.dataset}"
    if args.enable_thinking:
        env_tag += "_thinking"
    if args.prompt_variant != "official":
        env_tag += f"_prompt-{args.prompt_variant}"
    run_name = f"{env_tag}_{args.backend}_{args.model.replace('/', '_')}_{datetime.now():%Y%m%d_%H%M%S}"
    if args.tag:
        run_name += f"_{args.tag}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"env={args.env} ({env.description}), {len(questions)} puzzles, {args.rounds} rounds -> {out_dir}")

    samples = [{"question": q, "weak_demos": [], "output": []} for q in questions]
    summary = []
    for round_idx in range(args.rounds):
        prompts = [build_prompt(few_shot, s, round_idx, args) for s in samples]
        t0 = time.time()
        generations = generate(prompts)

        totals, solved = [], []
        for s, prompt, (gen, reasoning) in zip(samples, prompts, generations):
            answer = extract_answer(gen)
            res = env.evaluate(s["question"], answer)
            step_strs = [f"{x:.2f}" for x in res["step_rewards"]]
            s["weak_demos"].append({
                "prompt": s["question"],
                "answer": answer,
                "step_reward_strs": step_strs,
                "return_str": f"{res['total']:.2f}",
            })
            s["output"].append({
                "round": round_idx,
                "prompt": prompt,
                "generated_text": gen,
                "reasoning": reasoning,  # thinking text (not put back into the prompt)
                "answer": answer,
                "step_rewards": res["step_rewards"],
                "final_reward": res["final_reward"],
                "total": res["total"],
                "solved": res["solved"],
                "final_value": res["final_value"],
                "debug_log": res["debug_log"],  # analysis only, never shown to the model
            })
            totals.append(res["total"])
            solved.append(res["solved"])

        stats = {
            "round": round_idx,
            "mode": "exploit" if round_idx % 2 == 1 else "explore",
            "mean_reward": float(np.mean(totals)),
            "solve_rate": float(np.mean(solved)),
            "prompt_chars_max": max(len(p) for p in prompts),
            "seconds": round(time.time() - t0, 1),
        }
        summary.append(stats)
        best_so_far = np.mean([any(o["solved"] for o in s["output"]) for s in samples])
        print(f"round {round_idx:3d} [{stats['mode']:7s}] mean reward {stats['mean_reward']:6.2f} / 19  "
              f"solve {stats['solve_rate']:.1%}  solved-at-least-once {best_so_far:.1%}  "
              f"({stats['seconds']}s, longest prompt {stats['prompt_chars_max']} chars)")

        with open(out_dir / "outputs.json", "w") as f:
            json.dump([{"question": s["question"], "output": s["output"]} for s in samples], f, indent=1)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"done. round 0 (no rewards in context yet) mean reward {summary[0]['mean_reward']:.2f}, "
          f"last round {summary[-1]['mean_reward']:.2f}. Results in {out_dir}")


if __name__ == "__main__":
    main()
