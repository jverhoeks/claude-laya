"""Turn one Claude Code session on disk into a flat dict of counts and flags.

A session is `<project>/<id>.jsonl` plus any `<project>/<id>/subagents/*.jsonl`.
Parsing follows claudecounter: only real user prompts count, tool_use ids are
matched to their results, streamed duplicate assistant messages collapse, and
subagent files fold into the parent. `findings()` adds structural flags
(waste, loops, sprawl, routing) that need no model.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Text Claude Code injects under the user role; not something the user typed.
INJECTED = (
    "<task-notification>", "<command-name>", "<command-message>", "<command-args>",
    "<local-command-stdout>", "<local-command-stderr>", "<system-reminder>",
    "<user-prompt-submit-hook>", "<local-command-caveat>", "Caveat: The messages below",
)
TOOL_TARGET_KEYS = ("command", "file_path", "path", "pattern", "url", "query", "skill", "prompt", "description")
EDIT_TOOLS = ("Edit", "Write", "NotebookEdit")
# same starting points as claudecounter DefaultThresholds
REPEAT_TOOL_N, LOOP_MIN, READ_DUP_N = 3, 3, 2
SPRAWL_PROMPTS, SPRAWL_HOURS = 60, 4.0
ROUTING_MAX_TOKENS, ROUTING_MAX_TOOLS = 20_000, 5


@dataclass
class Transcript:
    """Everything accumulated while streaming through the jsonl lines."""
    session_id: str
    cwd: str | None = None
    start: str | None = None
    end: str | None = None
    prompts: list[str] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)   # {name, target, err, sub}
    turns: list[dict] = field(default_factory=list)   # {model, sub}
    tok_in: int = 0
    tok_out: int = 0
    tok_think: int = 0
    peak_context: int = 0
    cost_usd: float | None = None
    lines_added: int | None = None
    lines_removed: int | None = None
    has_pr: bool = False
    modes: Counter = field(default_factory=Counter)
    efforts: Counter = field(default_factory=Counter)
    seen_msg: set = field(default_factory=set)        # "msgid:requestid" already counted
    seen_tool: dict = field(default_factory=dict)     # tool_use id -> index in tools


# ---------------------------------------------------------------- small helpers

def parse_ts(raw) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(raw) if isinstance(raw, str) else None
    except ValueError:
        return None


def hours_between(start: str | None, end: str | None) -> float:
    a, b = parse_ts(start), parse_ts(end)
    return max(0.0, (b - a).total_seconds() / 3600) if a and b else 0.0


def clean_prompt(content) -> str:
    """User content -> one line of what the user actually typed, or ''."""
    if isinstance(content, list):
        content = "\n".join(b.get("text") or "" for b in content if isinstance(b, dict) and b.get("type") == "text")
    if not isinstance(content, str):
        return ""
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", content, flags=re.S)
    text = re.sub(r"<system-reminder>.*", "", text, flags=re.S).strip()  # unterminated tag
    if not text or text.startswith(INJECTED):
        return ""
    return re.sub(r"\s+", " ", text)[:800]


def tool_target(inp) -> str:
    if not isinstance(inp, dict):
        return ""
    return next((v.strip() for k in TOOL_TARGET_KEYS if isinstance(v := inp.get(k), str) and v.strip()), "")


# ---------------------------------------------------------------- one jsonl line

def apply_line(o: dict, t: Transcript, sub: bool) -> None:
    ts = o.get("timestamp")
    if isinstance(ts, str):
        t.start = min(t.start or ts, ts)
        t.end = max(t.end or ts, ts)
    if o.get("type") == "pr-link":
        t.has_pr = True
    if not sub:
        _note_session_meta(o, t)
    msg = o.get("message")
    if not isinstance(msg, dict):
        return
    if o.get("type") == "user" and not sub and not (o.get("isMeta") or o.get("isSidechain") or o.get("isCompactSummary")):
        if text := clean_prompt(msg.get("content")):
            t.prompts.append(text)
    _note_usage(o, msg, t, sub)
    _note_tools(msg.get("content"), t, sub)


def _note_session_meta(o: dict, t: Transcript) -> None:
    t.cwd = t.cwd or o.get("cwd")
    t.session_id = o.get("sessionId") or o.get("session_id") or t.session_id
    if pm := o.get("permissionMode") or o.get("permission-mode"):
        t.modes[pm] += 1
    if e := o.get("perTurnEffort") or o.get("effort"):
        t.efforts[str(e)] += 1
    if o.get("type") == "cost-state":
        if isinstance(c := o.get("totalCostUSD"), (int, float)):
            t.cost_usd = round(c, 4)
        t.lines_added = o.get("totalLinesAdded")
        t.lines_removed = o.get("totalLinesRemoved")


def _note_usage(o: dict, msg: dict, t: Transcript, sub: bool) -> None:
    usage, model = msg.get("usage"), msg.get("model")
    if not isinstance(usage, dict) or not model or model == "<synthetic>":
        return
    mid, rid = msg.get("id"), o.get("requestId")
    if mid and rid:  # streamed messages repeat the same ids; count once
        if (mid, rid) in t.seen_msg:
            return
        t.seen_msg.add((mid, rid))
    inn = int(usage.get("input_tokens") or 0)
    cc = int(usage.get("cache_creation_input_tokens") or 0)
    cr = int(usage.get("cache_read_input_tokens") or 0)
    details = usage.get("output_tokens_details")
    t.tok_in += inn
    t.tok_out += int(usage.get("output_tokens") or 0)
    t.tok_think += int(details.get("thinking_tokens") or 0) if isinstance(details, dict) else 0
    t.peak_context = max(t.peak_context, inn + cc + cr)
    t.turns.append({"model": model, "sub": sub})


def _note_tools(content, t: Transcript, sub: bool) -> None:
    if not isinstance(content, list):
        return
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "tool_use":
            bid = b.get("id") or ""
            if bid in t.seen_tool:
                continue
            if bid:
                t.seen_tool[bid] = len(t.tools)
            t.tools.append({"name": b.get("name") or "tool", "target": tool_target(b.get("input")), "err": False, "sub": sub})
        elif b.get("type") == "tool_result":
            i = t.seen_tool.get(b.get("tool_use_id") or "")
            if i is not None:
                t.tools[i]["err"] = bool(b.get("is_error"))


# ---------------------------------------------------------------- structural flags

def best_loop(seq: list[str]) -> dict | None:
    """Longest-covering cycle of 1-4 steps repeated >= LOOP_MIN times back to back."""
    best = None  # (cover, reps, w, at)
    for w in range(1, 5):
        for i in range(len(seq) - w + 1):
            reps, j = 1, i + w
            while j + w <= len(seq) and seq[i:i + w] == seq[j:j + w]:
                reps, j = reps + 1, j + w
            if reps >= LOOP_MIN and (best is None or reps * w > best[0]):
                best = (reps * w, reps, w, i)
    if not best:
        return None
    _, reps, w, at = best
    return {"detail": "cycle [%s] repeated %d×" % (" → ".join(seq[at:at + w]), reps), "count": reps}


def findings(t: Transcript, shape: str, model: str) -> list[dict]:
    out = []
    flag = lambda category, detail, count: out.append({"category": category, "detail": detail, "count": count})
    tools = t.tools

    if failed := sum(t_["err"] for t_ in tools):
        flag("waste", "%d failed tool call(s)" % failed, failed)

    reads = Counter(x["target"] for x in tools if x["name"] == "Read" and x["target"])
    dup = {f: n for f, n in reads.items() if n >= READ_DUP_N}
    if dup:
        extra = sum(dup.values()) - len(dup)
        flag("waste", "%d redundant Read(s) across %d file(s)" % (extra, len(dup)), extra)

    calls = Counter((x["name"], x["target"]) for x in tools if x["target"])
    (name, target), n = calls.most_common(1)[0] if calls else (("", ""), 0)
    if n >= REPEAT_TOOL_N:
        flag("abuse", "%s %r called %d×" % (name, target[:60], n), n)

    for stream, is_sub in (("main", False), ("subagent", True)):
        if loop := best_loop([x["name"] + ":" + x["target"] for x in tools if x["sub"] == is_sub]):
            flag("loop", "%s stream: %s" % (stream, loop["detail"]), loop["count"])

    hours = hours_between(t.start, t.end)
    if len(t.prompts) >= SPRAWL_PROMPTS or hours >= SPRAWL_HOURS:
        flag("sprawl", "long session: %d prompts over %.1fh" % (len(t.prompts), hours), len(t.prompts))

    if "opus" in model.lower() and t.tok_in + t.tok_out < ROUTING_MAX_TOKENS and len(tools) <= ROUTING_MAX_TOOLS:
        flag("routing", "light session ran on Opus", 1)

    n_read = sum(x["name"] == "Read" for x in tools)
    n_edit = sum(x["name"] in EDIT_TOOLS for x in tools)
    if len(tools) >= 15 and n_edit == 0 and not ((t.lines_added or 0) + (t.lines_removed or 0)):
        flag("idle", "many tools, no file edits", len(tools))
    elif shape == "explore":
        flag("explore", "explore-heavy: %d reads vs %d edits" % (n_read, n_edit), n_read)
    return out


def session_shape(n_tools: int, n_read: int, n_edit: int) -> str:
    if n_tools < 3:
        return "talk"
    if n_read >= 8 and n_read >= 3 * max(n_edit, 1):
        return "explore"
    if n_edit and n_edit >= n_read:
        return "ship"
    return "mixed"


# ---------------------------------------------------------------- entry point

def read_jsonl(path: Path):
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def parse_session(path: Path) -> dict | None:
    """Main transcript + folded subagents -> flat record, or None if nothing happened."""
    t = Transcript(session_id=path.stem)
    for o in read_jsonl(path):
        apply_line(o, t, sub=False)
    for p in sorted((path.with_suffix("") / "subagents").glob("*.jsonl")):
        for o in read_jsonl(p):
            apply_line(o, t, sub=True)
    if not t.prompts or not t.turns:
        return None

    models = Counter(x["model"] for x in t.turns)
    model = models.most_common(1)[0][0]
    names = Counter(x["name"] for x in t.tools)
    n_read, n_edit = names["Read"], sum(names[n] for n in EDIT_TOOLS)
    shape = session_shape(len(t.tools), n_read, n_edit)
    started = parse_ts(t.start)
    info = path.stat()
    return {
        "path": str(path),
        "session_id": t.session_id,
        "project": t.cwd.rstrip("/").split("/")[-1] if t.cwd else path.parent.name,
        "cwd": t.cwd,
        "started": t.start,
        "ended": t.end,
        "hours": round(hours_between(t.start, t.end), 2),
        "hour": started.hour if started else None,
        "mtime": int(info.st_mtime),
        "size": info.st_size,
        "model": model,
        "models": [m for m, _ in models.most_common()],
        "effort": t.efforts.most_common(1)[0][0] if t.efforts else None,
        "n_prompts": len(t.prompts),
        "n_assistant": len(t.turns),
        "n_tools": len(t.tools),
        "n_errors": sum(x["err"] for x in t.tools),
        "n_sub_tools": sum(x["sub"] for x in t.tools),
        "n_read": n_read, "n_edit": n_edit, "n_bash": names["Bash"], "n_skill": names["Skill"],
        "shape": shape,
        "n_bypass": t.modes["bypassPermissions"],
        "tool_top": [n for n, _ in names.most_common(4)],
        "tok_in": t.tok_in, "tok_out": t.tok_out, "tok_think": t.tok_think,
        "peak_context": t.peak_context,
        "cost_usd": t.cost_usd,
        "lines_added": t.lines_added, "lines_removed": t.lines_removed,
        "has_pr": t.has_pr,
        "findings": findings(t, shape, model),
        "first_prompt": t.prompts[0][:400],
        "last_prompt": t.prompts[-1][:240] if len(t.prompts) > 1 else None,
    }
