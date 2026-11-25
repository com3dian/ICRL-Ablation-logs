"""Game of 24 with hidden rules.

The model gets the ORIGINAL Game of 24 prompt (game24/game24_3steps.txt +
original instructions): it believes it is playing normal decimal Game of 24 with
target "24". The environment secretly uses different rules and only returns
rewards, so the model can discover the rules only through trial and reward.

Environments (see ENVS):
    ops_swap : + and - swapped, * and / swapped ("a + b" really is a - b, ...)
    base3    : puzzle numbers are written in base 3; target is the true value 24
               ("24" cannot be written in base 3, so the prompt's 24 stays decimal)
    base7    : puzzle numbers AND the target "24" are base 7 -> true target 18

How a response is scored (same reward structure as the original game24 ICRL):
    **Response** Step1: 4 + 6 = 10 (left: 10 2 3) Step2: ... Step3: ... **Answer**: ...
  - Each step: the env takes the two numbers the model picked and the symbol it
    used, and computes the TRUE result under the hidden rules. The result the
    model wrote ("10") is kept only as a label, so a later step that says "10"
    refers to the true result of this step.
  - Step reward: 3 if the TRUE remaining numbers can still reach the target, else 0.
  - Final reward: 10 if after 3 valid steps the TRUE value equals the target.
  - The `**Answer**:` line is ignored.
Rewards are computed exactly (Fraction arithmetic + brute-force solver); an LLM
judge cannot be used because it would judge by normal arithmetic.
"""
import json
import random
import re
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement, permutations
from pathlib import Path

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
NORMAL_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
}

SWAPPED_OPS = {  # "相反": + <-> -, * <-> /
    "+": NORMAL_OPS["-"],
    "-": NORMAL_OPS["+"],
    "*": NORMAL_OPS["/"],
    "/": NORMAL_OPS["*"],
}

ENVS = {
    "ops_swap": dict(ops=SWAPPED_OPS, base=10, target=24,
                     description="+ means minus, - means plus, * means divide, / means multiply"),
    "base3": dict(ops=NORMAL_OPS, base=3, target=24,
                  description="numbers written in base 3; target is the true value 24"),
    "base7": dict(ops=NORMAL_OPS, base=7, target=int("24", 7),
                  description="numbers and target '24' written in base 7; true target 18"),
    # Control: no hidden rule. Use with another env's puzzles, e.g. --env normal --dataset ops_swap
    "normal": dict(ops=NORMAL_OPS, base=10, target=24,
                   description="control: normal Game of 24 (no hidden rule)"),
}
HIDDEN_RULE_ENVS = [name for name in ENVS if name != "normal"]

SHOWN_TARGET = 24  # what the prompt says (and what the model believes)

STEP_REWARD_SOLVABLE = 3.0   # valid step, TRUE remaining numbers can still reach the target
STEP_REWARD_DEAD_END = 0.0   # valid step, but the target is no longer reachable
STEP_REWARD_INVALID = 0.0    # unparseable / number not available / undefined (e.g. divide by 0)
FINAL_REWARD_CORRECT = 10.0
FINAL_REWARD_WRONG = 0.0
NUM_STEPS = 3

DATA_DIR = Path(__file__).parent / "data"

_SYMBOL_ALIASES = {"×": "*", "÷": "/", "−": "-", "–": "-"}
_NUM = r"-?\d+(?:\.\d+)?(?:/\d+)?"
_STEP_RE = re.compile(rf"({_NUM})\s*([-+*/])\s*({_NUM})\s*=\s*({_NUM})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_base(n: int, base: int) -> str:
    if n == 0:
        return "0"
    sign, n = ("-", -n) if n < 0 else ("", n)
    digits = []
    while n:
        n, r = divmod(n, base)
        digits.append(str(r))
    return sign + "".join(reversed(digits))


def fmt(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def _apply(ops, sym, a, b):
    if a is None or b is None:
        return None
    try:
        return Fraction(ops[sym](a, b))
    except (ZeroDivisionError, ValueError, OverflowError):
        return None


def _take(pool, label):
    """pool entries are [label, true_value]. Remove the entry the model calls
    `label` (newest first) and return its true value, or None."""
    for k in range(len(pool) - 1, -1, -1):
        if pool[k][0] == label:
            return pool.pop(k)[1]
    return None


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class HiddenRuleGame24Env:
    def __init__(self, name, ops=NORMAL_OPS, base=10, target=24, description=""):
        assert 2 <= base <= 10
        self.name = name
        self.ops = ops
        self.base = base
        self.target = Fraction(target)
        self.description = description
        self._solve = lru_cache(maxsize=None)(self._solve_impl)

    # ---- numbers ---------------------------------------------------------
    def labels(self, question: str):
        """What the model believes the numbers are (decimal reading)."""
        return [Fraction(int(s)) for s in question.split()]

    def true_values(self, question: str):
        return [Fraction(int(s, self.base)) for s in question.split()]

    def display(self, values) -> str:
        return " ".join(to_base(int(v), self.base) for v in values)

    # ---- solver (hidden rules) -------------------------------------------
    def _solve_impl(self, nums: tuple):
        if len(nums) == 1:
            return () if nums[0] == self.target else None
        for i, j in permutations(range(len(nums)), 2):
            rest = [nums[k] for k in range(len(nums)) if k not in (i, j)]
            for sym in self.ops:
                r = _apply(self.ops, sym, nums[i], nums[j])
                if r is None:
                    continue
                sub = self._solve(tuple(sorted(rest + [r])))
                if sub is not None:
                    return ((nums[i], sym, nums[j], r),) + sub
        return None

    def solve_values(self, values):
        """Steps (a, sym, b, result) in TRUE values, or None if unsolvable."""
        return self._solve(tuple(sorted(Fraction(v) for v in values)))

    def find_solution(self, question: str):
        """Human-readable solution in TRUE values (for analysis only)."""
        steps = self.solve_values(self.true_values(question))
        if steps is None:
            return None
        return [f"{fmt(a)} {s} {fmt(b)} = {fmt(r)}" for a, s, b, r in steps]

    def is_solvable(self, question: str) -> bool:
        return self.solve_values(self.true_values(question)) is not None

    # ---- reward ----------------------------------------------------------
    def evaluate(self, question: str, response: str) -> dict:
        """Execute the model's plan under the hidden rules and score it.

        Returns:
            step_rewards: list of NUM_STEPS floats (missing steps padded with 0)
            final_reward, total, solved
            final_value: TRUE final value (str) or None
            debug_log:   per-step explanation with TRUE values.
                         FOR ANALYSIS ONLY - never put it into the model's prompt.
        """
        m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        text = m.group(1) if m else response
        if "**Response**" in text:
            text = text.split("**Response**", 1)[1]
        text = text.split("**Answer**", 1)[0]
        for k, v in _SYMBOL_ALIASES.items():
            text = text.replace(k, v)

        pool = [[lab, tru] for lab, tru in zip(self.labels(question), self.true_values(question))]
        steps = _STEP_RE.findall(text)[:NUM_STEPS]

        step_rewards, debug_log = [], []
        valid = True
        for a_s, sym, b_s, c_s in steps:
            desc = f"{a_s} {sym} {b_s} = {c_s}"
            if not valid:
                step_rewards.append(STEP_REWARD_INVALID)
                debug_log.append(f"{desc}: skipped (previous step invalid)")
                continue
            trial = [list(p) for p in pool]
            a, b = _take(trial, Fraction(a_s)), _take(trial, Fraction(b_s))
            if a is None or b is None:
                valid = False
                step_rewards.append(STEP_REWARD_INVALID)
                debug_log.append(f"{desc}: uses a number that is not available")
                continue
            true_r = _apply(self.ops, sym, a, b)
            if true_r is None:
                valid = False
                step_rewards.append(STEP_REWARD_INVALID)
                debug_log.append(f"{desc}: undefined (true {fmt(a)} {sym} {fmt(b)})")
                continue
            pool = trial + [[Fraction(c_s), true_r]]
            true_left = [p[1] for p in pool]
            reachable = self.solve_values(true_left) is not None
            step_rewards.append(STEP_REWARD_SOLVABLE if reachable else STEP_REWARD_DEAD_END)
            debug_log.append(
                f"{desc}: true {fmt(a)} {sym} {fmt(b)} = {fmt(true_r)}, "
                f"true left {[fmt(x) for x in true_left]}, "
                f"{'target reachable' if reachable else 'dead end'}")

        if len(steps) < NUM_STEPS:
            debug_log.append(f"only {len(steps)} parseable step(s)")
        step_rewards += [STEP_REWARD_INVALID] * (NUM_STEPS - len(step_rewards))

        done = valid and len(steps) == NUM_STEPS and len(pool) == 1
        final_value = pool[0][1] if done else None
        solved = final_value == self.target
        final_reward = FINAL_REWARD_CORRECT if solved else FINAL_REWARD_WRONG
        return {
            "step_rewards": step_rewards,
            "final_reward": final_reward,
            "total": sum(step_rewards) + final_reward,
            "solved": solved,
            "final_value": fmt(final_value) if final_value is not None else None,
            "debug_log": debug_log,
        }

    # ---- dataset construction ----------------------------------------------
    def normal_reading_check(self, question: str):
        """Simulate a model that believes normal decimal arithmetic.

        Returns (decimal_solvable, leaky):
            decimal_solvable: the puzzle, read as normal decimal 24, has a solution
                              (so the model can write a correct-looking answer)
            leaky:            some such correct-looking plan ALSO solves the puzzle
                              under the hidden rules (puzzle doesn't test the rule)
        """
        found = False

        def rec(pool):  # pool: list of (label, true)
            nonlocal found
            if len(pool) == 1:
                lab, tru = pool[0]
                if lab == SHOWN_TARGET:
                    found = True
                    return tru is not None and tru == self.target
                return False
            for i, j in permutations(range(len(pool)), 2):
                rest = [pool[k] for k in range(len(pool)) if k not in (i, j)]
                for sym in NORMAL_OPS:
                    lab = _apply(NORMAL_OPS, sym, pool[i][0], pool[j][0])
                    if lab is None:
                        continue
                    tru = _apply(self.ops, sym, pool[i][1], pool[j][1])
                    if rec(rest + [(lab, tru)]):
                        return True
            return False

        leaky = rec(list(zip(self.labels(question), self.true_values(question))))
        return found, leaky

    def build_dataset(self, n=100, max_value=13, seed=0):
        """Fixed puzzle set. True values are drawn from 1..max_value (same range as
        the original 24.csv). Kept puzzles are:
            1. solvable under the hidden rules
            2. solvable when (mis)read as normal decimal Game of 24, so the model
               confidently writes a correct-looking answer
            3. not leaky: no correct-looking normal plan also works under the hidden rules
        """
        combos = list(combinations_with_replacement(range(1, max_value + 1), 4))
        random.Random(seed).shuffle(combos)
        stats = {"examined": 0, "unsolvable": 0, "decimal_unsolvable": 0, "leaky": 0}
        puzzles = []
        for combo in combos:
            if len(puzzles) >= n:
                break
            stats["examined"] += 1
            q = self.display(combo)
            if not self.is_solvable(q):
                stats["unsolvable"] += 1
                continue
            decimal_solvable, leaky = self.normal_reading_check(q)
            if not decimal_solvable:
                stats["decimal_unsolvable"] += 1
                continue
            if leaky:
                stats["leaky"] += 1
                continue
            puzzles.append({
                "id": len(puzzles),
                "question": q,
                "true_values": [int(v) for v in combo],
                "example_solution": self.find_solution(q),  # analysis only, never shown to the model
            })
        stats["kept"] = len(puzzles)
        return puzzles, stats


def make_env(name: str) -> HiddenRuleGame24Env:
    return HiddenRuleGame24Env(name, **ENVS[name])


def load_dataset(name: str):
    """List of question strings for env `name` (built by build_datasets.py)."""
    with open(DATA_DIR / f"{name}.json") as f:
        return [p["question"] for p in json.load(f)["puzzles"]]


# ---------------------------------------------------------------------------
# Self-test / examples
# ---------------------------------------------------------------------------
def _response(steps_text):
    return ("<answer>**Response** "
            + " ".join(f"Step{k + 1}: {s}" for k, s in enumerate(steps_text))
            + " **Answer**: -</answer>")


def _normal_plan(question):
    """Plan a normal-arithmetic model would write (labels, decimal reading)."""
    normal = HiddenRuleGame24Env("normal")
    steps = normal.solve_values([int(s) for s in question.split()])
    return None if steps is None else [f"{fmt(a)} {s} {fmt(b)} = {fmt(r)}" for a, s, b, r in steps]


def _true_plan(env, question):
    """The true solution written the way a model that figured out the rules would."""
    pool = [[lab, tru] for lab, tru in zip(env.labels(question), env.true_values(question))]
    out = []
    for k, (a, sym, b, r) in enumerate(env.solve_values(env.true_values(question))):
        labels = []
        for v in (a, b):
            idx = max(i for i, p in enumerate(pool) if p[1] == v)
            labels.append(pool.pop(idx)[0])
        r_label = (Fraction(int(to_base(int(r), env.base))) if r.denominator == 1 and r >= 0
                   else Fraction(900 + k))  # base-N can't write fractions/negatives; any fresh label works
        pool.append([r_label, r])
        out.append(f"{fmt(labels[0])} {sym} {fmt(labels[1])} = {fmt(r_label)}")
    return out


if __name__ == "__main__":
    for name in ENVS:
        env = make_env(name)
        puzzles, _ = env.build_dataset(n=2, seed=123)
        print(f"===== {name}: {env.description} =====")
        for p in puzzles:
            q = p["question"]
            print(f"puzzle '{q}'  true values {p['true_values']}  true target {fmt(env.target)}")
            for tag, plan in (("normal-thinking", _normal_plan(q)), ("knows-the-rule", _true_plan(env, q))):
                r = env.evaluate(q, _response(plan))
                print(f"  {tag:16s}: {' | '.join(plan)}  -> {r['step_rewards']} + {r['final_reward']} = {r['total']}")
        print()
