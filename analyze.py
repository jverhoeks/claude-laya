#!/usr/bin/env python3
"""Score Claude Code sessions with a local Laya model.

    uv sync
    uv run python analyze.py --limit 50     # smoke
    uv run python analyze.py                # all main sessions
    uv run python analyze.py --days 30

Flow per session: transcript.parse_session -> laya_judge.card -> Laya.ask
-> laya_judge.judge -> one line in claude-laya-out/sessions.jsonl.
Parse runs in threads; Laya answers serially (one GPU). Re-runs skip files
whose size/mtime/schema have not changed unless --force.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from laya_judge import SCHEMA, Laya, card, judge
from transcript import parse_session


def find_files(root: Path, days: int) -> list[Path]:
    """Main transcripts only, newest first. agent-* files are folded in by the parser."""
    cutoff = time.time() - days * 86400 if days else 0
    files = [p for p in root.glob("*/*.jsonl") if not p.name.startswith("agent-") and p.stat().st_mtime >= cutoff]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def skip_key(session_id, size, mtime, schema=SCHEMA) -> str:
    return "%s:%s:%s:%s" % (session_id, size, mtime, schema)


def timed_parse(path: Path) -> tuple[dict | None, float]:
    t0 = time.perf_counter()
    return parse_session(path), round((time.perf_counter() - t0) * 1000, 1)


def pctile(xs: list[float], p: int) -> float:
    xs = sorted(xs)
    return xs[int(p / 100 * (len(xs) - 1))]


def fmt_ms(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    return "p50 %.0f  p95 %.0f  max %.0f  n=%d" % (pctile(xs, 50), pctile(xs, 95), max(xs), len(xs))


def write_resume(rows: list[dict], dest: Path) -> str:
    n = len(rows)

    def mix(key, value=lambda r, k: r.get(k) or "?"):
        c = Counter(value(r, key) for r in rows)
        return ", ".join("%.0f%% %s" % (100 * v / n, k) for k, v in c.most_common()) if n and c else "n/a"

    daypart = lambda r, _: (
        "night" if r["hour"] < 6 else "morning" if r["hour"] < 12 else "afternoon" if r["hour"] < 18 else "evening"
    ) if isinstance(r.get("hour"), int) else "?"
    flags = Counter(c for r in rows for c in {f["category"] for f in r.get("findings") or []})  # per session
    vague = sum((r.get("prompt_clear") or 1) < 0.4 for r in rows)
    rules = sum((r.get("standing_rule") or 0) >= 0.55 for r in rows)
    parse_ms = [r["timing"]["parse_ms"] for r in rows if r.get("timing")]
    laya_ms = [r["timing"]["laya_ms"] for r in rows if r.get("timing")]
    lines = [
        "# Claude session check — %d sessions · %s" % (n, dt.date.today().isoformat()),
        "",
        "- Model fit: " + mix("model_fit", lambda r, k: r.get(k) or "about_right"),
        "- Session: " + mix("session_ok", lambda r, k: "ok" if r.get(k) else "issues"),
        "- Intent: " + mix("intent"),
        "- Outcome: " + mix("outcome"),
        "- Shape: " + mix("shape"),
        "- Hardness: " + mix("hardness"),
        "- Flags: " + (", ".join("%.0f%% %s" % (100 * v / n, k) for k, v in flags.most_common()) or "none"),
        "- Models: " + mix("model"),
        "- When: " + mix("hour", daypart),
        "- Vague first prompt: %.0f%% · standing rule: %.0f%%" % (100 * vague / n if n else 0, 100 * rules / n if n else 0),
        "",
        "- parse %s" % fmt_ms(parse_ms),
        "- laya  %s" % fmt_ms(laya_ms),
        "",
    ]
    text = "\n".join(lines)
    dest.write_text(text + "\n", encoding="utf-8")
    return text


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path.home() / ".claude" / "projects")
    ap.add_argument("--out", type=Path, default=Path("claude-laya-out"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--days", type=int, default=0, help="only sessions modified in the last N days")
    ap.add_argument("--parse-workers", type=int, default=4, help="jsonl parse threads; Laya stays serial (one GPU)")
    ap.add_argument("--force", action="store_true", help="rescore everything")
    args = ap.parse_args(argv)
    if not args.dir.is_dir():
        sys.exit("no such dir: %s" % args.dir)

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl_path, resume_path = args.out / "sessions.jsonl", args.out / "resume.md"

    rows: dict[str, dict] = {}  # session_id -> scored record
    if jsonl_path.exists() and not args.force:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                rows[r["session_id"]] = r
        print("[skip] %d already scored" % len(rows), flush=True)
    done = {skip_key(r["session_id"], r["size"], r["mtime"], r.get("schema")) for r in rows.values()}

    files = find_files(args.dir, args.days)[: args.limit or None]
    todo = [p for p in files if skip_key(p.stem, p.stat().st_size, int(p.stat().st_mtime)) not in done]
    print("[scan] %d files, %d to score, %d skipped" % (len(files), len(todo), len(files) - len(todo)), flush=True)

    laya = None
    scored = empty = 0
    t0 = time.perf_counter()
    # ponytail: pool.map parses everything eagerly; add a bounded queue if memory ever matters
    with ThreadPoolExecutor(max_workers=max(1, args.parse_workers)) as pool, \
            jsonl_path.open("w" if args.force else "a", encoding="utf-8") as sink:
        for s, parse_ms in pool.map(timed_parse, todo):
            if s is None:
                empty += 1
                continue
            if skip_key(s["session_id"], s["size"], s["mtime"]) in done:
                continue  # sessionId inside the file differs from the filename; already scored
            laya = laya or Laya()
            text = card(s)
            answers, laya_ms = laya.ask(text)
            rec = judge(s, answers)
            rec["card"] = text
            rec["timing"] = {"parse_ms": parse_ms, "laya_ms": laya_ms}
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sink.flush()
            rows[rec["session_id"]] = rec
            scored += 1
            if scored % 10 == 0:
                print("  %d scored in %.0fs" % (scored, time.perf_counter() - t0), flush=True)

    wall = time.perf_counter() - t0
    jsonl_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values()), encoding="utf-8")
    print("\n" + write_resume(list(rows.values()), resume_path))
    print("wall %.1fs · %d scored · %d empty · %.1f/s" % (wall, scored, empty, scored / wall if wall else 0))
    print("wrote %s (%d sessions) and %s" % (jsonl_path, len(rows), resume_path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
