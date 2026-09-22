"""Run the cards already in sessions.jsonl through both Laya backends and show where they differ.

    uv run python compare_backends.py [--out claude-laya-out]
"""
import json
import sys
from pathlib import Path

from laya_judge import Laya, judge

CHOICE = ("hardness", "intent", "outcome", "model_fit", "session_ok")
NOUL = ("friction", "needs_reasoning", "prompt_clear", "standing_rule")


def main(argv: list[str]) -> int:
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else Path("claude-laya-out")
    rows = [json.loads(l) for l in (out / "sessions.jsonl").open()]
    if not rows:
        print("no sessions scored yet; run analyze.py first")
        return 1
    scored = {}
    for backend in ("mlx", "torch"):
        laya = Laya(backend)
        scored[backend] = []
        for r in rows:
            answers, ms = laya.ask(r["card"])
            scored[backend].append((judge(r, answers), ms))
        del laya

    diffs = 0
    for r, (a, _), (b, _) in zip(rows, scored["mlx"], scored["torch"]):
        for k in CHOICE:
            if a[k] != b[k]:
                diffs += 1
                print("%s %-15s mlx=%-14s torch=%s" % (r["session_id"][:8], k, a[k], b[k]))
        for k in NOUL:
            if abs(a[k] - b[k]) > 0.05:
                diffs += 1
                print("%s %-15s mlx=%.3f torch=%.3f" % (r["session_id"][:8], k, a[k], b[k]))
    print("\n%d sessions, %d differences" % (len(rows), diffs))
    for backend in ("mlx", "torch"):
        ms = sorted(m for _, m in scored[backend])
        print("%-5s p50 %4.0f ms  max %4.0f ms" % (backend, ms[len(ms) // 2], ms[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
