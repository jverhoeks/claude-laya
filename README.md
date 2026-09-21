# claude-laya

Score your local Claude Code sessions with [Laya](https://github.com/NandhaKishorM/laya): was the model too cheap / about right / too expensive, and was the session ok.

Standalone. No playground, no HTTP server. `uv sync` installs Laya; the script loads the english checkpoint in-process.

Reads `~/.claude/projects/*/*.jsonl`. Parsing follows [claudecounter](https://github.com/jverhoeks/claudecounter) (real prompts only, tool ids, subagents folded in). Laya labels a short card per session. Counts stay in code.

## Run

```bash
uv sync
uv run python analyze.py --limit 50          # smoke
uv run python analyze.py                     # all main sessions
uv run python analyze.py --days 30
```

Python 3.12+. First run downloads open weights (~2.3 GB) into `~/.cache/huggingface` unless they are already there. Hugging Face’s native xet client is disabled (it can stall at 0 bytes); TensorFlow is not imported.

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

Predict is one GPU at a time. Parse is threaded (`--parse-workers 4`).
