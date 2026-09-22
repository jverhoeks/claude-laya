"""The Laya part: ask a small local model what a session was like.

Laya never sees the raw transcript. `card()` writes a short plain-English
description from the parsed counts, Laya answers QUESTIONS about that card,
and `judge()` turns the answers plus the structural findings into
`model_fit` (too_cheap / about_right / too_expensive) and `session_ok`.

Loads in-process by default. Set LAYA_API=http://127.0.0.1:8770 to reuse a
Laya HTTP server that is already running (two loaded models fight over the GPU).
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request

from transcript import FAIL_MIN, FAIL_RATE

# Must be set before `import laya` (done lazily in Laya.__init__).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # native xet client can stall at 0 bytes
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# LAYA_BACKEND=mlx (Apple Silicon default, loads in <1s) or torch (upstream package).
BACKEND = os.environ.get("LAYA_BACKEND") or ("mlx" if (sys.platform, platform.machine()) == ("darwin", "arm64") else "torch")
REPO = {"mlx": "aac6fef/laya-mlx", "torch": "convaiinnovations/laya"}


def hf_cache(backend: str) -> str:
    return os.path.expanduser("~/.cache/huggingface/hub/models--%s/snapshots" % REPO[backend].replace("/", "--"))


if os.path.isdir(hf_cache(BACKEND)):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

CHECKPOINT = os.environ.get("LAYA_CHECKPOINT", "english")
SCHEMA = "v3"  # bump to rescore everything when QUESTIONS or judge() change

QUESTIONS = {
    "hardness": {
        "type": "choice",
        "instructions": "How hard is the user's request as stated?",
        "criteria": {
            "mechanical": "rename, format, git, config tweak, one-file copy or a small known edit",
            "standard": "implement a feature, fix a bug, or review a clear target",
            "hard": "architecture, tricky debugging, security, or an unclear multi-step design",
        },
    },
    "intent": {
        "type": "choice",
        "instructions": "What is the user asking the coding agent to do?",
        "criteria": {
            "implement": "write or change code to add something",
            "fix": "debug or repair something broken",
            "review": "review, audit, or check existing work",
            "explain": "explain or discuss, no edit asked",
            "chore": "git, config, rename, setup, housekeeping",
            "other": "none of the above",
        },
    },
    "outcome": {
        "type": "choice",
        "instructions": "How does this session read as having ended?",
        "criteria": {
            "done": "the user accepts the work or asks to commit/ship",
            "abandoned": "the user stops while still blocked or unhappy",
            "switched": "the user changes task or says they will continue later",
            "unclear": "the ending is not stated",
        },
    },
    "friction": {
        "type": "noul",
        "instructions": "Does this write-up describe user frustration, corrections, loops, or repeated failures?",
        "criteria": {
            "true": "the user is correcting the agent, errors keep happening, or the work is looping",
            "false": "the session proceeds without the user fighting the agent",
        },
    },
    "needs_reasoning": {
        "type": "noul",
        "instructions": "Did the task as stated need a strong reasoning model?",
        "criteria": {
            "true": "needs design, tradeoffs, or hard diagnosis",
            "false": "a cheaper coding model could follow the instructions",
        },
    },
    "prompt_clear": {
        "type": "noul",
        "instructions": "Does the first user request name a concrete target (file, bug, or change)?",
        "criteria": {
            "true": "specific enough to act on",
            "false": "vague, open-ended, or missing the target",
        },
    },
    "standing_rule": {
        "type": "noul",
        "instructions": "Is the user restating a standing project rule or preference they should not have to repeat?",
        "criteria": {
            "true": "a convention, style, or constraint restated as if the agent forgot",
            "false": "a one-off task request",
        },
    },
}

# model name substring -> price tier (0 cheap, 1 default, 2 frontier)
TIER = {"haiku": 0, "sonnet": 1, "fable": 1, "opus": 2}
TIER_WORD = {0: "a cheap small model", 1: "a mid-size default model", 2: "an expensive frontier model"}


def model_tier(name: str) -> int:
    name = (name or "").lower()
    return next((tier for needle, tier in TIER.items() if needle in name), 1)


# ---------------------------------------------------------------- the card

def few_many(n: int) -> str:
    """Laya reads words better than digits."""
    if n <= 0:
        return "no"
    if n <= 2:
        return "a couple of"
    if n <= 6:
        return "several"
    if n <= 20:
        return "many"
    return "a large number of"


def card(s: dict) -> str:
    """Plain-English write-up of a parsed session; this is all Laya sees."""
    lines = ["The user asked: " + s["first_prompt"]]
    if s.get("last_prompt"):
        lines.append("Later they said: " + s["last_prompt"])
    lines.append("The agent used %s (%s). Effort was %s."
                 % (TIER_WORD[model_tier(s["model"])], s["model"], s.get("effort") or "unspecified"))
    if len(s.get("models") or []) > 1:
        lines.append("The session switched models: " + ", ".join(s["models"]) + ".")
    top = " (" + ", ".join(s["tool_top"]) + ")" if s["tool_top"] else ""
    lines.append("The session had %s user requests, %s assistant turns, %s tool calls%s."
                 % (few_many(s["n_prompts"]), few_many(s["n_assistant"]), few_many(s["n_tools"]), top))
    lines.append("It looked like a %s session (reads %d, edits %d)." % (s["shape"], s["n_read"], s["n_edit"]))
    lines.append("There were %s tool errors." % few_many(s["n_errors"]))
    for f in s["findings"][:4]:
        lines.append("Observed: " + f["detail"] + ".")
    changed = (s["lines_added"] or 0) + (s["lines_removed"] or 0)
    if changed:
        lines.append("A small number of lines were changed." if changed < 30 else
                     "A moderate amount of code was changed." if changed < 300 else
                     "A large amount of code was changed.")
    if s["cost_usd"] is not None:
        lines.append("The API cost was %s." % ("low" if s["cost_usd"] < 1 else "noticeable" if s["cost_usd"] < 10 else "high"))
    return "\n".join(lines)


# ---------------------------------------------------------------- the model

class Laya:
    """One loaded checkpoint. `ask(card)` returns (answers, milliseconds)."""

    def __init__(self, backend: str = BACKEND):
        self.backend = backend
        self.api = os.environ.get("LAYA_API")
        self.agent = None
        if self.api and self._server_ready():
            print("[laya] using %s (%s ready)" % (self.api, CHECKPOINT), flush=True)
            return
        if self.api:
            print("[laya] LAYA_API=%s is not ready, loading in-process" % self.api, flush=True)
        self.api = None
        self._load_local()

    def _server_ready(self) -> bool:
        try:
            health = json.load(urllib.request.urlopen(self.api + "/api/health", timeout=3))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return False
        return (health.get("models") or {}).get(CHECKPOINT) == "ready"

    def _load_local(self) -> None:
        cache = hf_cache(self.backend)
        if os.path.isdir(cache):
            print("[laya] loading weights from %s (mlx: ~1s, torch: 30-90s, no progress bar)" % cache, flush=True)
        else:
            print("[laya] first run: downloading ~2.3 GB of weights", flush=True)
        t0 = time.perf_counter()
        laya = importlib.import_module("laya_mlx" if self.backend == "mlx" else "laya")

        print("[laya] imported %s %s in %.1fs, building %s checkpoint..."
              % (laya.__name__, laya.__version__, time.perf_counter() - t0, CHECKPOINT), flush=True)
        t1 = time.perf_counter()
        self.agent = laya.load(REPO[self.backend])
        self.ask("Warm up. The user asked to rename a variable in one file.")
        print("[laya] ready on %s in %.1fs" % (self.agent.device, time.perf_counter() - t1), flush=True)

    def ask(self, state: str) -> tuple[dict, float]:
        t0 = time.perf_counter()
        answers = self._ask_http(state) if self.api else self.agent.predict(state, QUESTIONS)["answers"]
        return answers, round((time.perf_counter() - t0) * 1000, 1)

    def _ask_http(self, state: str) -> dict:
        body = json.dumps({"state": state, "questions": QUESTIONS, "model": CHECKPOINT}).encode()
        req = urllib.request.Request(self.api + "/api/predict", body, {"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                return json.load(urllib.request.urlopen(req, timeout=120))["answers"]
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                err = e
                time.sleep(0.5 + attempt)
        raise RuntimeError("Laya HTTP kept failing: %s" % err)


# ---------------------------------------------------------------- the verdict

def _choice(answers: dict, key: str, default: str) -> tuple[str, float]:
    a = answers.get(key) or {}
    pick = a.get("choice") or default
    return pick, round(float((a.get("probabilities") or {}).get(pick, 0) or 0), 3)


def _noul(answers: dict, key: str) -> float:
    return round(float((answers.get(key) or {}).get("noul") or 0), 3)


def model_fit(s: dict, hardness: str, needs_reasoning: float) -> str:
    tier = model_tier(s["model"])
    cats = {f["category"] for f in s["findings"]}
    if "routing" in cats or (tier >= 2 and hardness == "mechanical" and needs_reasoning < 0.4):
        return "too_expensive"
    struggled = s["n_errors"] >= 3 or "loop" in cats
    if tier <= 0 and (hardness == "hard" or (hardness == "standard" and needs_reasoning >= 0.7 and struggled)):
        return "too_cheap"
    return "about_right"


def judge(s: dict, answers: dict) -> dict:
    """Parsed session + Laya answers -> the scored record written to sessions.jsonl."""
    hardness, hardness_p = _choice(answers, "hardness", "standard")
    intent, intent_p = _choice(answers, "intent", "other")
    outcome, outcome_p = _choice(answers, "outcome", "unclear")
    friction = _noul(answers, "friction")
    needs_reasoning = _noul(answers, "needs_reasoning")

    cats = {f["category"] for f in s["findings"]}
    issues = []
    if friction >= 0.65:  # noul scores cluster at 0.5±0.1; only a clear yes counts
        issues.append("friction")
    if s["n_errors"] >= FAIL_MIN and s["n_errors"] >= FAIL_RATE * s["n_tools"]:
        issues.append("tool_errors")
    if "loop" in cats:
        issues.append("loop")
    if outcome == "abandoned":
        issues.append("abandoned")

    return dict(
        s,
        schema=SCHEMA,
        model_tier=model_tier(s["model"]),
        hardness=hardness, hardness_p=hardness_p,
        intent=intent, intent_p=intent_p,
        outcome=outcome, outcome_p=outcome_p,
        friction=friction,
        needs_reasoning=needs_reasoning,
        prompt_clear=_noul(answers, "prompt_clear"),
        standing_rule=_noul(answers, "standing_rule"),
        model_fit=model_fit(s, hardness, needs_reasoning),
        session_ok=not issues,
        issues=issues,
    )
