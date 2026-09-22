"""Smallest check that fails if parsing or the verdict logic breaks. Run: uv run python test_analyze.py"""
import tempfile
from pathlib import Path

from analyze import pctile
from laya_judge import card, judge, model_fit
from transcript import parse_session

MAIN = """\
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
SUB = """\
{"type":"assistant","timestamp":"2026-06-01T10:00:30Z","isSidechain":true,"requestId":"req9","message":{"id":"msg9","model":"claude-haiku-4-5","usage":{"input_tokens":7,"output_tokens":3},"content":[{"type":"tool_use","id":"tu9","name":"Read","input":{"file_path":"/tmp/proj/go.mod"}}]}}
{"type":"user","timestamp":"2026-06-01T10:00:31Z","isSidechain":true,"permissionMode":"bypassPermissions","message":{"content":[{"type":"tool_result","tool_use_id":"tu9","content":"module x"}]}}
"""
ANSWERS = {
    "hardness": {"choice": "standard", "probabilities": {"standard": 0.8}},
    "intent": {"choice": "implement", "probabilities": {"implement": 0.7}},
    "outcome": {"choice": "abandoned", "probabilities": {"abandoned": 0.6}},
    "friction": {"noul": 0.9},
    "needs_reasoning": {"noul": 0.5},
    "prompt_clear": {"noul": 0.3},
    "standing_rule": {"noul": 0.1},
}

tmp = Path(tempfile.mkdtemp())
main = tmp / "abc123.jsonl"
main.write_text(MAIN, encoding="utf-8")
(tmp / "abc123" / "subagents").mkdir(parents=True)
(tmp / "abc123" / "subagents" / "agent-1.jsonl").write_text(SUB, encoding="utf-8")

s = parse_session(main)
assert s, "parse returned None"
assert s["session_id"] == "s1"
assert s["n_prompts"] == 2, s["n_prompts"]
assert s["first_prompt"] == "please do X" and s["last_prompt"] == "keep this part"
assert s["n_tools"] == 3, s["n_tools"]          # duplicate streamed Edit collapsed; sub Read folded in
assert s["n_errors"] == 1 and s["n_sub_tools"] == 1
assert s["n_assistant"] == 3, s["n_assistant"]  # msg1 + msg2 + sub, not the streamed dup
assert s["tok_in"] == 117 and s["peak_context"] == 3100
assert (s["n_read"], s["n_edit"], s["n_bash"]) == (1, 1, 1)
assert s["shape"] == "ship" and s["n_bypass"] == 1
assert [f["category"] for f in s["findings"]] == ["routing"], s["findings"]  # 1 of 3 failed is not waste

assert model_fit(s, "standard", 0.5) == "too_expensive"                    # routing flag
assert model_fit(dict(s, model="claude-haiku-4-5", findings=[]), "hard", 0.2) == "too_cheap"
assert model_fit(dict(s, model="claude-sonnet-5", findings=[]), "standard", 0.5) == "about_right"

c = card(s)
assert "please do X" in c and "expensive frontier" in c and "Observed: light session ran on Opus." in c

rec = judge(s, ANSWERS)
assert rec["model_fit"] == "too_expensive" and rec["model_tier"] == 2
assert rec["issues"] == ["friction", "abandoned"] and rec["session_ok"] is False
assert rec["hardness_p"] == 0.8 and rec["prompt_clear"] == 0.3

assert pctile([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 50) == 50
print("ok")
