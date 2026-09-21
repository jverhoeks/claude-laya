#!/usr/bin/env python3
"""Score Claude Code JSONL sessions with the local Laya model.

Parser and structural flags follow claudecounter/claudeinsights (real-prompt
filter, tool_use id matching, subagent fold, loops, light-Opus routing).
Laya judges a short card (hardness, friction, intent, outcome, …). Counts stay in code.

    uv sync
    uv run python analyze.py --limit 50
    uv run python analyze.py

Loads Laya in-process (no playground). Set LAYA_API=http://127.0.0.1:8770 only
if you already have a Laya HTTP server running. Parse is threaded
(--parse-workers, default 4). Predict is serial: one GPU lock. Resume is
percentages plus parse/model/rtt p50/p95.

Writes claude-laya-out/sessions.jsonl and resume.md (gitignored). Re-runs skip
unchanged files unless --force. Main transcripts only; agent-* files are folded
in from <session>/subagents/, not scored on their own.
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # native xet client can stall at 0 bytes
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_CACHE = os.path.expanduser("~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots")
if os.path.isdir(_CACHE):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import datetime as dt
import http.client
import json
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Thread

LAYA_API = os.environ.get("LAYA_API")  # unset → load in-process; do not assume a playground server
CHECKPOINT = os.environ.get("LAYA_CHECKPOINT", "english")

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
SCHEMA = "v2"  # bump to rescore when questions/findings change

TIER = [("haiku", 0), ("sonnet", 1), ("fable", 1), ("opus", 2)]
TIER_WORD = {0: "a cheap small model", 1: "a mid-size default model", 2: "an expensive frontier model"}
INJECTED = (
    "<task-notification>", "<command-name>", "<command-message>", "<command-args>",
    "<local-command-stdout>", "<local-command-stderr>", "<system-reminder>",
    "<user-prompt-submit-hook>", "<local-command-caveat>",
)
TOOL_TARGET_KEYS = ("command", "file_path", "path", "pattern", "url", "query", "skill", "prompt", "description")
# same starting points as claudecounter DefaultThresholds
REPEAT_TOOL_N, LOOP_MIN, READ_DUP_N = 3, 3, 2
SPRAWL_PROMPTS, SPRAWL_HOURS = 60, 4.0
ROUTING_MAX_TOKENS, ROUTING_MAX_TOOLS = 20_000, 5


def model_tier(name: str) -> int:
    n = (name or "").lower()
    if not n or n == "<synthetic>":
        return 1
    for needle, t in TIER:
        if needle in n:
            return t
    return 1


def pctile(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[int(p / 100 * (len(xs) - 1))]


def fmt_ms(xs) -> str:
    if not xs:
        return "n/a"
    return "p50 %.0f  p95 %.0f  max %.0f  n=%d" % (pctile(xs, 50), pctile(xs, 95), max(xs), len(xs))


def few_many(n: int) -> str:
    if n <= 0:
        return "no"
    if n <= 2:
        return "a couple of"
    if n <= 6:
        return "several"
    if n <= 20:
        return "many"
    return "a large number of"


def strip_reminders(text: str) -> str:
    open_, close = "<system-reminder>", "</system-reminder>"
    while True:
        i = text.find(open_)
        if i < 0:
            return text
        j = text.find(close, i)
        if j < 0:
            return text[:i]
        text = text[:i] + text[j + len(close):]


def is_injected(text: str) -> bool:
    s = text.lstrip()
    return s.startswith(INJECTED) or s.startswith("Caveat: The messages below")


def prompt_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b.get("text") or "" for b in content if isinstance(b, dict) and b.get("type") == "text")


def tool_target(inp) -> str:
    if not isinstance(inp, dict):
        return ""
    for k in TOOL_TARGET_KEYS:
        v = inp.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _ts(raw) -> str | None:
    if not raw:
        return None
    if isinstance(raw, str):
        return raw
    return None


def parse_file(path: Path, st: dict, sub: bool) -> None:
    try:
        fh = path.open(encoding="utf-8")
    except OSError:
        return
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            apply_line(o, st, sub)


def apply_line(o: dict, st: dict, sub: bool) -> None:
    ts = _ts(o.get("timestamp"))
    if ts:
        if st["start"] is None or ts < st["start"]:
            st["start"] = ts
        if st["end"] is None or ts > st["end"]:
            st["end"] = ts
    if not sub:
        st["cwd"] = st["cwd"] or o.get("cwd")
        st["session_id"] = o.get("sessionId") or o.get("session_id") or st["session_id"]
        pm = o.get("permissionMode") or o.get("permission-mode")
        if pm:
            st["modes"][pm] += 1
        e = o.get("perTurnEffort") or o.get("effort")
        if e:
            st["efforts"][str(e)] += 1
    if o.get("type") == "pr-link":
        st["has_pr"] = True
    if o.get("type") == "cost-state" and not sub:
        c = o.get("totalCostUSD")
        if isinstance(c, (int, float)):
            st["cost_usd"] = round(c, 4)
        st["lines_added"] = o.get("totalLinesAdded")
        st["lines_removed"] = o.get("totalLinesRemoved")

    msg = o.get("message") if isinstance(o.get("message"), dict) else None
    if o.get("type") == "user" and not sub and not o.get("isMeta") and not o.get("isSidechain") and not o.get("isCompactSummary") and msg:
        text = strip_reminders(prompt_text(msg.get("content"))).strip()
        if text and not is_injected(text):
            st["prompts"].append(re.sub(r"\s+", " ", text)[:800])

    if not msg:
        return

    usage, model = msg.get("usage"), msg.get("model")
    if isinstance(usage, dict) and model and model != "<synthetic>":
        mid, rid = msg.get("id") or "", o.get("requestId") or ""
        key = "%s:%s" % (mid, rid) if mid and rid else None
        if key is None or key not in st["seen_msg"]:
            if key:
                st["seen_msg"].add(key)
            inn = int(usage.get("input_tokens") or 0)
            out = int(usage.get("output_tokens") or 0)
            cc = int(usage.get("cache_creation_input_tokens") or 0)
            cr = int(usage.get("cache_read_input_tokens") or 0)
            think = 0
            details = usage.get("output_tokens_details")
            if isinstance(details, dict):
                think = int(details.get("thinking_tokens") or 0)
            st["tok_in"] += inn
            st["tok_out"] += out
            st["tok_cc"] += cc
            st["tok_think"] += think
            st["peak"] = max(st["peak"], inn + cc + cr)
            st["turns"].append({"model": model, "sub": sub, "in": inn, "out": out})

    content = msg.get("content")
    if not isinstance(content, list):
        return
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "tool_use":
            bid = b.get("id") or ""
            if bid:
                if bid in st["seen_tool"]:
                    continue
                st["seen_tool"][bid] = len(st["tools"])
            st["tools"].append({
                "name": b.get("name") or "tool",
                "target": tool_target(b.get("input")),
                "err": False, "sub": sub,
            })
        elif b.get("type") == "tool_result":
            i = st["seen_tool"].get(b.get("tool_use_id") or "")
            if i is not None:
                st["tools"][i]["err"] = bool(b.get("is_error"))


def findings(st: dict) -> list[dict]:
    out = []
    tools = st["tools"]
    failed = sum(1 for t in tools if t["err"])
    if failed:
        out.append({"category": "waste", "detail": "%d failed tool call(s)" % failed, "count": failed})
    reads = Counter(t["target"] for t in tools if t["name"] == "Read" and t["target"])
    extra = files = 0
    for n in reads.values():
        if n >= READ_DUP_N:
            extra += n - 1
            files += 1
    if extra:
        out.append({"category": "waste", "detail": "%d redundant Read(s) across %d file(s)" % (extra, files), "count": extra})
    counts = Counter((t["name"], t["target"]) for t in tools if t["target"])
    abuse = [(k, n) for k, n in counts.items() if n >= REPEAT_TOOL_N]
    if abuse:
        name, tgt = max(abuse, key=lambda kv: kv[1])[0]
        n = max(abuse, key=lambda kv: kv[1])[1]
        out.append({"category": "abuse", "detail": "%s %r called %d×" % (name, tgt[:60], n), "count": n})
    for stream, seq in (
        ("main", [t["name"] + ":" + t["target"] for t in tools if not t["sub"]]),
        ("subagent", [t["name"] + ":" + t["target"] for t in tools if t["sub"]]),
    ):
        loop = best_loop(seq)
        if loop:
            out.append({"category": "loop", "detail": "%s stream: %s" % (stream, loop["detail"]), "count": loop["count"]})
    n_prompts = len(st["prompts"])
    hours = 0.0
    if st["start"] and st["end"] and st["end"] > st["start"]:
        try:
            a, b = dt.datetime.fromisoformat(st["start"].replace("Z", "+00:00")), dt.datetime.fromisoformat(st["end"].replace("Z", "+00:00"))
            hours = (b - a).total_seconds() / 3600
        except ValueError:
            hours = 0.0
    if n_prompts >= SPRAWL_PROMPTS or hours >= SPRAWL_HOURS:
        out.append({"category": "sprawl", "detail": "long session: %d prompts over %.1fh" % (n_prompts, hours), "count": n_prompts})
    models = Counter(t["model"] for t in st["turns"])
    model = models.most_common(1)[0][0] if models else ""
    work = st["tok_in"] + st["tok_out"]
    if "opus" in model.lower() and work < ROUTING_MAX_TOKENS and len(tools) <= ROUTING_MAX_TOOLS:
        out.append({"category": "routing", "detail": "light session ran on Opus", "count": 1})
    n_read = sum(1 for t in tools if t["name"] == "Read")
    n_edit = sum(1 for t in tools if t["name"] in ("Edit", "Write", "NotebookEdit"))
    lines = (st.get("lines_added") or 0) + (st.get("lines_removed") or 0)
    if len(tools) >= 15 and n_edit == 0 and lines == 0:
        out.append({"category": "idle", "detail": "many tools, no file edits", "count": len(tools)})
    elif n_read >= 8 and n_read >= 3 * max(n_edit, 1):
        out.append({"category": "explore", "detail": "explore-heavy: %d reads vs %d edits" % (n_read, n_edit), "count": n_read})
    return out


def best_loop(seq: list[str]) -> dict | None:
    best_reps, best_cover, best_w, best_at = 0, 0, 0, 0
    for w in range(1, 5):
        for i in range(0, len(seq) - w + 1):
            reps = 1
            j = i + w
            while j + w <= len(seq) and seq[i:i + w] == seq[j:j + w]:
                reps += 1
                j += w
            if reps >= LOOP_MIN and reps * w > best_cover:
                best_reps, best_cover, best_w, best_at = reps, reps * w, w, i
    if best_reps < LOOP_MIN:
        return None
    cycle = " → ".join(seq[best_at:best_at + best_w])
    return {"detail": "cycle [%s] repeated %d×" % (cycle, best_reps), "count": best_reps}


def parse_session(path: Path) -> dict | None:
    st = {
        "session_id": path.stem, "cwd": None, "start": None, "end": None,
        "prompts": [], "tools": [], "turns": [], "seen_msg": set(), "seen_tool": {},
        "tok_in": 0, "tok_out": 0, "tok_cc": 0, "tok_think": 0, "peak": 0,
        "cost_usd": None, "lines_added": None, "lines_removed": None, "has_pr": False,
        "modes": Counter(), "efforts": Counter(),
    }
    parse_file(path, st, False)
    subdir = path.with_suffix("") / "subagents"
    if subdir.is_dir():
        for p in sorted(subdir.glob("*.jsonl")):
            parse_file(p, st, True)
    if not st["prompts"] or not st["turns"]:
        return None
    models = Counter(t["model"] for t in st["turns"])
    model = models.most_common(1)[0][0]
    names = Counter(t["name"] for t in st["tools"])
    n_read, n_edit = names.get("Read", 0), names.get("Edit", 0) + names.get("Write", 0)
    n_bash, n_skill = names.get("Bash", 0), names.get("Skill", 0)
    n_tools = len(st["tools"])
    if n_tools < 3:
        shape = "talk"
    elif n_read >= 8 and n_read >= 3 * max(n_edit, 1):
        shape = "explore"
    elif n_edit >= n_read and n_edit:
        shape = "ship"
    else:
        shape = "mixed"
    hours = session_hours(st["start"], st["end"])
    hour = None
    if st["start"]:
        try:
            hour = dt.datetime.fromisoformat(st["start"].replace("Z", "+00:00")).hour
        except ValueError:
            hour = None
    info = path.stat()
    cwd = st["cwd"] or ""
    project = cwd.rstrip("/").split("/")[-1] if cwd else path.parent.name
    return {
        "path": str(path),
        "session_id": st["session_id"],
        "project": project,
        "cwd": st["cwd"],
        "started": st["start"],
        "ended": st["end"],
        "hours": round(hours, 2),
        "hour": hour,
        "mtime": int(info.st_mtime),
        "size": info.st_size,
        "schema": SCHEMA,
        "model": model,
        "models": [m for m, _ in models.most_common()],
        "model_tier": model_tier(model),
        "effort": st["efforts"].most_common(1)[0][0] if st["efforts"] else None,
        "n_prompts": len(st["prompts"]),
        "n_assistant": len(st["turns"]),
        "n_tools": n_tools,
        "n_errors": sum(1 for t in st["tools"] if t["err"]),
        "n_sub_tools": sum(1 for t in st["tools"] if t["sub"]),
        "n_read": n_read, "n_edit": n_edit, "n_bash": n_bash, "n_skill": n_skill,
        "shape": shape,
        "n_bypass": st["modes"].get("bypassPermissions", 0),
        "tool_top": [n for n, _ in names.most_common(4)],
        "tok_in": st["tok_in"], "tok_out": st["tok_out"], "tok_think": st["tok_think"],
        "peak_context": st["peak"],
        "cost_usd": st["cost_usd"],
        "lines_added": st["lines_added"], "lines_removed": st["lines_removed"],
        "has_pr": st["has_pr"],
        "findings": findings(st),
        "first_prompt": st["prompts"][0][:400],
        "last_prompt": st["prompts"][-1][:240] if len(st["prompts"]) > 1 else None,
    }


def session_hours(start, end) -> float:
    if not start or not end or end <= start:
        return 0.0
    try:
        a = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
        b = dt.datetime.fromisoformat(end.replace("Z", "+00:00"))
        return max(0.0, (b - a).total_seconds() / 3600)
    except ValueError:
        return 0.0


def model_fit(s: dict, hardness: str, needs_reasoning: float) -> str:
    cats = {f["category"] for f in s["findings"]}
    if "routing" in cats or (s["model_tier"] >= 2 and hardness == "mechanical" and needs_reasoning < 0.4):
        return "too_expensive"
    struggled = s["n_errors"] >= 3 or "loop" in cats
    if s["model_tier"] <= 0 and (hardness == "hard" or (hardness == "standard" and needs_reasoning >= 0.7 and struggled)):
        return "too_cheap"
    return "about_right"


def card(s: dict) -> str:
    bits = ["The user asked: " + s["first_prompt"]]
    if s.get("last_prompt"):
        bits.append("Later they said: " + s["last_prompt"])
    bits.append("The agent used %s (%s). Effort was %s." % (TIER_WORD[s["model_tier"]], s["model"], s.get("effort") or "unspecified"))
    if len(s.get("models") or []) > 1:
        bits.append("The session switched models: " + ", ".join(s["models"]) + ".")
    bits.append(
        "The session had %s user requests, %s assistant turns, %s tool calls%s."
        % (few_many(s["n_prompts"]), few_many(s["n_assistant"]), few_many(s["n_tools"]),
           (" (" + ", ".join(s["tool_top"]) + ")" if s["tool_top"] else ""))
    )
    bits.append("It looked like a %s session (reads %d, edits %d)." % (s.get("shape") or "mixed", s.get("n_read") or 0, s.get("n_edit") or 0))
    bits.append("There were %s tool errors." % few_many(s["n_errors"]))
    for f in s["findings"][:4]:
        bits.append("Observed: " + f["detail"] + ".")
    if s["lines_added"] or s["lines_removed"]:
        n = (s["lines_added"] or 0) + (s["lines_removed"] or 0)
        bits.append("No lines of code were changed." if n == 0 else
                    "A small number of lines were changed." if n < 30 else
                    "A moderate amount of code was changed." if n < 300 else
                    "A large amount of code was changed.")
    if s["cost_usd"] is not None:
        bits.append("The API cost was %s." % ("low" if s["cost_usd"] < 1 else "noticeable" if s["cost_usd"] < 10 else "high"))
    return "\n".join(bits)


def _choice(answers, key, default="unclear"):
    a = answers.get(key) or {}
    pick = a.get("choice") or default
    probs = a.get("probabilities") or {}
    return pick, round(float(probs.get(pick, 0) or 0), 3)


def _noul(answers, key):
    a = answers.get(key) or {}
    return round(float(a.get("noul") or 0), 3)


def score_answers(s: dict, answers: dict) -> dict:
    hardness, hp = _choice(answers, "hardness", "standard")
    intent, intent_p = _choice(answers, "intent", "other")
    outcome, outcome_p = _choice(answers, "outcome", "unclear")
    friction = _noul(answers, "friction")
    needs = _noul(answers, "needs_reasoning")
    prompt_clear = _noul(answers, "prompt_clear")
    standing_rule = _noul(answers, "standing_rule")
    fit = model_fit(s, hardness, needs)
    cats = {f["category"] for f in s["findings"]}
    issues = []
    if friction >= 0.5:
        issues.append("friction")
    if s["n_errors"] >= 5:
        issues.append("tool_errors")
    if "loop" in cats:
        issues.append("loop")
    if outcome == "abandoned":
        issues.append("abandoned")
    out = dict(s)
    out.update(
        hardness=hardness, hardness_p=hp,
        intent=intent, intent_p=intent_p,
        outcome=outcome, outcome_p=outcome_p,
        friction=friction, needs_reasoning=needs,
        prompt_clear=prompt_clear, standing_rule=standing_rule,
        model_fit=fit, session_ok=not issues, issues=issues,
    )
    return out


class Laya:
    def __init__(self):
        self.mode = None
        self.conn = None
        self.agent = None
        self._http_or_local()

    def _http_or_local(self):
        if LAYA_API:
            u = urllib.parse.urlparse(LAYA_API)
            try:
                health = json.load(urllib.request.urlopen(LAYA_API + "/api/health", timeout=3))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
                health = None
            if health and (health.get("models") or {}).get(CHECKPOINT) == "ready":
                self.mode = "http"
                self.conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=120)
                print("[laya] using %s (%s ready)" % (LAYA_API, CHECKPOINT), flush=True)
                return
            print("[laya] LAYA_API=%s is not ready, loading in-process" % LAYA_API, flush=True)
        else:
            print("[laya] loading weights in-process (cache hit: 30-90s, no progress bar)", flush=True)
        if os.path.isdir(_CACHE):
            print("[laya] using Hugging Face cache at %s" % _CACHE, flush=True)
        else:
            print("[laya] first run: downloading ~2.3 GB of weights (HF_HUB_DISABLE_XET=1)", flush=True)
        t0 = time.perf_counter()
        import laya
        print("[laya] imported laya %s in %.1fs, building english checkpoint..." % (laya.__version__, time.perf_counter() - t0), flush=True)
        t1 = time.perf_counter()
        self.agent = laya.load("convaiinnovations/laya")
        self.mode = "local"
        print("[laya] ready on %s in %.1fs" % (self.agent.device, time.perf_counter() - t1), flush=True)

    def predict(self, state: str) -> tuple[dict, dict]:
        t0 = time.perf_counter()
        if self.mode == "http":
            body = json.dumps({"state": state, "questions": QUESTIONS, "model": CHECKPOINT}).encode()
            for attempt in range(3):
                try:
                    if self.conn is None:
                        u = urllib.parse.urlparse(LAYA_API)
                        self.conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=120)
                    self.conn.request("POST", "/api/predict", body, {"Content-Type": "application/json"})
                    r = self.conn.getresponse()
                    data = r.read()
                    if r.status != 200:
                        raise RuntimeError("Laya HTTP %s: %s" % (r.status, data[:300]))
                    res = json.loads(data)
                    rtt = (time.perf_counter() - t0) * 1000
                    return res["answers"], {"rtt_ms": round(rtt, 1), "model_ms": res.get("latency_ms")}
                except (OSError, http.client.HTTPException):
                    self.conn = None
                    t0 = time.perf_counter()
                    time.sleep(0.5 + attempt)
            raise RuntimeError("Laya HTTP kept failing")
        answers = self.agent.predict(state, QUESTIONS)["answers"]
        ms = round((time.perf_counter() - t0) * 1000, 1)
        return answers, {"rtt_ms": ms, "model_ms": ms}


def iter_files(root: Path, days: int) -> list[Path]:
    cutoff = time.time() - days * 86400 if days else 0
    files = []
    for p in root.glob("*/*.jsonl"):
        if p.name.startswith("agent-"):
            continue
        if cutoff and p.stat().st_mtime < cutoff:
            continue
        files.append(p)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def skip_key(r: dict) -> str:
    return "%s:%s:%s:%s" % (r.get("session_id"), r.get("size"), r.get("mtime"), r.get("schema") or SCHEMA)


def write_resume(rows: list[dict], dest: Path) -> str:
    n = len(rows)
    fit, ok, hard, models, cats = Counter(), Counter(), Counter(), Counter(), Counter()
    intent, outcome, shape, when = Counter(), Counter(), Counter(), Counter()
    for r in rows:
        fit[r.get("model_fit") or "about_right"] += 1
        ok["ok" if r.get("session_ok") else "issues"] += 1
        hard[r.get("hardness") or "?"] += 1
        models[r.get("model") or "?"] += 1
        intent[r.get("intent") or "?"] += 1
        outcome[r.get("outcome") or "?"] += 1
        shape[r.get("shape") or "?"] += 1
        h = r.get("hour")
        if isinstance(h, int):
            when["night" if h < 6 else "morning" if h < 12 else "afternoon" if h < 18 else "evening"] += 1
        for f in r.get("findings") or []:
            cats[f["category"]] += 1
    def mix(c):
        if not n or not c:
            return "n/a"
        return ", ".join("%.0f%% %s" % (100 * v / n, k) for k, v in c.most_common())

    vague = sum(1 for r in rows if (r.get("prompt_clear") or 1) < 0.4)
    rules = sum(1 for r in rows if (r.get("standing_rule") or 0) >= 0.55)
    lines = [
        "# Claude session check — %d sessions · %s" % (n, dt.date.today().isoformat()),
        "",
        "- Model fit: " + mix(fit),
        "- Session: " + mix(ok),
        "- Intent: " + mix(intent),
        "- Outcome: " + mix(outcome),
        "- Shape: " + mix(shape),
        "- Hardness: " + mix(hard),
        "- Flags: " + (mix(cats) if cats else "none"),
        "- Models: " + mix(models),
        "- When: " + mix(when),
        "- Vague first prompt: %.0f%% · standing rule: %.0f%%" % (100 * vague / n if n else 0, 100 * rules / n if n else 0),
        "",
    ]
    parse_ms = [r["timing"]["parse_ms"] for r in rows if isinstance(r.get("timing"), dict) and r["timing"].get("parse_ms") is not None]
    model_ms = [r["timing"]["model_ms"] for r in rows if isinstance(r.get("timing"), dict) and r["timing"].get("model_ms") is not None]
    rtt_ms = [r["timing"]["rtt_ms"] for r in rows if isinstance(r.get("timing"), dict) and r["timing"].get("rtt_ms") is not None]
    if parse_ms or model_ms or rtt_ms:
        lines += [
            "- parse %s" % fmt_ms(parse_ms),
            "- model %s" % fmt_ms(model_ms),
            "- rtt   %s" % fmt_ms(rtt_ms),
            "",
        ]
    text = "\n".join(lines)
    dest.write_text(text + "\n", encoding="utf-8")
    return text


MAIN_FIXTURE = """\
{"type":"user","timestamp":"2026-06-01T10:00:00Z","sessionId":"s1","cwd":"/tmp/proj","permissionMode":"default","message":{"role":"user","content":"please do X"}}
{"type":"assistant","timestamp":"2026-06-01T10:00:05Z","requestId":"req1","message":{"id":"msg1","model":"claude-opus-4-7","usage":{"input_tokens":100,"output_tokens":50,"cache_creation_input_tokens":1000,"cache_read_input_tokens":2000},"content":[{"type":"text","text":"on it"},{"type":"tool_use","id":"tu1","name":"Bash","input":{"command":"go test ./..."}}]}}
{"type":"user","timestamp":"2026-06-01T10:00:09Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu1","content":"ok"}]}}
{"type":"user","timestamp":"2026-06-01T10:00:10Z","permissionMode":"default","message":{"content":"<task-notification> <task-id>b1</task-id>"}}
{"type":"user","timestamp":"2026-06-01T10:00:11Z","permissionMode":"default","isMeta":true,"message":{"content":"meta junk"}}
{"type":"user","timestamp":"2026-06-01T10:01:00Z","permissionMode":"bypassPermissions","message":{"content":"keep this<system-reminder>drop me</system-reminder> part"}}
{"type":"assistant","timestamp":"2026-06-01T10:01:05Z","requestId":"req2","message":{"id":"msg2","model":"claude-opus-4-7","usage":{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":100,"cache_read_input_tokens":200},"content":[{"type":"tool_use","id":"tu2","name":"Edit","input":{"file_path":"/tmp/proj/main.go"}}]}}
{"type":"assistant","timestamp":"2026-06-01T10:01:05Z","requestId":"req2","message":{"id":"msg2","model":"claude-opus-4-7","usage":{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":100,"cache_read_input_tokens":200},"content":[{"type":"tool_use","id":"tu2","name":"Edit","input":{"file_path":"/tmp/proj/main.go"}}]}}
{"type":"user","timestamp":"2026-06-01T10:01:09Z","message":{"content":[{"type":"tool_result","tool_use_id":"tu2","content":"no such file","is_error":true}]}}
"""
SUB_FIXTURE = """\
{"type":"assistant","timestamp":"2026-06-01T10:00:30Z","isSidechain":true,"requestId":"req9","message":{"id":"msg9","model":"claude-haiku-4-5","usage":{"input_tokens":7,"output_tokens":3},"content":[{"type":"tool_use","id":"tu9","name":"Read","input":{"file_path":"/tmp/proj/go.mod"}}]}}
{"type":"user","timestamp":"2026-06-01T10:00:31Z","isSidechain":true,"permissionMode":"bypassPermissions","message":{"content":[{"type":"tool_result","tool_use_id":"tu9","content":"module x"}]}}
"""


def self_check() -> int:
    tmp = Path(tempfile.mkdtemp())
    main = tmp / "abc123.jsonl"
    main.write_text(MAIN_FIXTURE, encoding="utf-8")
    sub = tmp / "abc123" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-1.jsonl").write_text(SUB_FIXTURE, encoding="utf-8")
    s = parse_session(main)
    assert s, "parse returned None"
    assert s["n_prompts"] == 2, s["n_prompts"]
    assert "please do X" in s["first_prompt"] and "keep this part" in (s["last_prompt"] or "")
    assert s["n_tools"] == 3, s["n_tools"]          # dup Edit collapsed; sub Read folded
    assert s["n_errors"] == 1, s["n_errors"]
    assert s["n_sub_tools"] == 1, s["n_sub_tools"]
    assert s["n_assistant"] == 3, s["n_assistant"]  # msg1 + msg2 + sub, not the streamed dup
    assert s["tok_in"] == 117, s["tok_in"]
    assert s["model_tier"] == 2
    assert s["shape"] in ("talk", "explore", "ship", "mixed")
    assert s["n_read"] == 1 and s["n_edit"] == 1 and s["n_bash"] == 1
    light = dict(s, findings=[{"category": "routing", "detail": "x", "count": 1}], model_tier=2)
    assert model_fit(light, "standard", 0.5) == "too_expensive"
    assert model_fit(dict(s, model_tier=0, findings=[], n_errors=0), "hard", 0.2) == "too_cheap"
    assert model_fit(dict(s, model_tier=1, findings=[], n_errors=0), "standard", 0.5) == "about_right"
    c = card(s)
    assert "please do X" in c and "expensive" in c
    sample = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert pctile(sample, 50) == 50 and pctile(sample, 95) == 90
    print("self-check ok")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path.home() / ".claude" / "projects")
    ap.add_argument("--out", type=Path, default=Path("claude-laya-out"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--days", type=int, default=0, help="only sessions modified in the last N days")
    ap.add_argument("--parse-workers", type=int, default=4, help="jsonl parse threads; predict stays serial (one GPU)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args(argv)
    if args.self_check:
        return self_check()
    if not args.dir.is_dir():
        sys.exit("no such dir: %s" % args.dir)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    jsonl_path, resume_path = out / "sessions.jsonl", out / "resume.md"
    done, rows = set(), []
    if jsonl_path.exists() and not args.force:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            r = json.loads(line)
            done.add(skip_key(r))
            rows.append(r)
        print("[skip] %d already scored" % len(done), flush=True)
    elif args.force and jsonl_path.exists():
        jsonl_path.unlink()

    files = iter_files(args.dir, args.days)
    if args.limit:
        files = files[: args.limit]
    todo = []
    for p in files:
        st = p.stat()
        if "%s:%s:%s:%s" % (p.stem, st.st_size, int(st.st_mtime), SCHEMA) in done and not args.force:
            continue
        todo.append(p)
    print("[scan] %d files, %d to score, %d skipped" % (len(files), len(todo), len(files) - len(todo)), flush=True)

    workers = max(1, args.parse_workers)
    q = Queue(maxsize=max(8, workers * 2))

    def parse_one(p: Path):
        t0 = time.perf_counter()
        s = parse_session(p)
        return s, round((time.perf_counter() - t0) * 1000, 1)

    def producer():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(parse_one, p) for p in todo]
            for f in futs:
                q.put(f.result())
        q.put(None)

    Thread(target=producer, daemon=True).start()
    laya = None
    scored = empty = 0
    parse_ms, model_ms, rtt_ms = [], [], []
    t0 = time.perf_counter()
    with jsonl_path.open("a", encoding="utf-8") as sink:
        while True:
            item = q.get()
            if item is None:
                break
            s, pms = item
            parse_ms.append(pms)
            if not s:
                empty += 1
                continue
            if skip_key(s) in done and not args.force:
                continue
            if laya is None:
                laya = Laya()
                laya.predict("Warm up. The user asked to rename a variable in one file.")
            text = card(s)
            answers, tpred = laya.predict(text)
            rec = score_answers(s, answers)
            rec["card"] = text
            rec["timing"] = {"parse_ms": pms, "rtt_ms": tpred.get("rtt_ms"), "model_ms": tpred.get("model_ms")}
            if tpred.get("model_ms") is not None:
                model_ms.append(float(tpred["model_ms"]))
            if tpred.get("rtt_ms") is not None:
                rtt_ms.append(float(tpred["rtt_ms"]))
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
            rows.append(rec)
            done.add(skip_key(rec))
            scored += 1
            if scored % 10 == 0:
                print(
                    "  %d scored  parse %s | model %s | rtt %s | %.0fs"
                    % (scored, fmt_ms(parse_ms), fmt_ms(model_ms), fmt_ms(rtt_ms), time.perf_counter() - t0),
                    flush=True,
                )

    wall = time.perf_counter() - t0
    by_id = {r["session_id"]: r for r in rows}
    rows = list(by_id.values())
    jsonl_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    text = write_resume(rows, resume_path)
    print("\n" + text)
    print("wall %.1fs · %d scored · %d empty · %.1f/s" % (wall, scored, empty, scored / wall if wall else 0))
    print("parse %s" % fmt_ms(parse_ms))
    print("model %s" % fmt_ms(model_ms))
    print("rtt   %s" % fmt_ms(rtt_ms))
    print("wrote %s (%d sessions) and %s" % (jsonl_path, len(rows), resume_path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
