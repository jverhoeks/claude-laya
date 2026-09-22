# claude-laya

Score your local Claude Code sessions with [Laya](https://github.com/NandhaKishorM/laya): was the model too cheap / about right / too expensive, and was the session ok.

Standalone. No playground, no HTTP server. `uv sync` installs Laya; the script loads the english checkpoint in-process.

Reads `~/.claude/projects/*/*.jsonl`.

- `transcript.py` parses one session into counts and structural flags, following [claudecounter](https://github.com/jverhoeks/claudecounter) (real prompts only, tool ids, subagents folded in).
- `laya_judge.py` is the Laya part: the questions, the plain-English card Laya reads, the model loader, and the verdict (`model_fit`, `session_ok`).
- `analyze.py` is the CLI: find files, parse in threads, ask Laya serially, write the outputs.

## Run

```bash
uv sync
uv run python analyze.py --limit 50          # smoke
uv run python analyze.py                     # all main sessions
uv run python analyze.py --days 30
uv run python test_analyze.py                 # self-check, no model needed
```

Backend: on Apple Silicon the default is [laya-mlx](https://github.com/mizorewww/laya-mlx) (independent MLX port, same weights, loads in about a second). Elsewhere, or with `LAYA_BACKEND=torch`, the upstream PyTorch package is used. To score the already-scored cards with both and print every disagreement plus latency:

```bash
uv run python compare_backends.py
```

Python 3.12+. First run downloads open weights (~0.9 GB mlx, ~2.3 GB torch) into `~/.cache/huggingface` unless they are already there. Hugging Face’s native xet client is disabled (it can stall at 0 bytes); TensorFlow is not imported.

Do not load this and another Laya process (for example the playground server) at the same time — they fight over the GPU. If a server is already up and you want to reuse it:

```bash
LAYA_API=http://127.0.0.1:8770 uv run python analyze.py
```

Writes `claude-laya-out/sessions.jsonl` and `claude-laya-out/resume.md` (gitignored; they contain your prompts). Re-runs skip unchanged files. `--force` rescores.

Resume is percentages, for example:

```
- Model fit: 81% about_right, 12% too_expensive, 7% too_cheap
- Session: 70% ok, 30% issues
- Intent: 40% implement, 22% fix, …
```

Laya answers one session at a time (one GPU). Parse is threaded (`--parse-workers 4`). The resume ends with parse and Laya latency percentiles.
