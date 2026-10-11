#!/usr/bin/env python3
"""Grade HumanEval completions. Run ONLY inside a network-less, throwaway container:

  docker run --rm --network none --memory 4g --pids-limit 256 -v RESULTS:/w:ro -v SCRIPTS:/s:ro \
      --entrypoint python3 lmsysorg/sglang:v0.5.21-cu130 /s/grade_humaneval.py /w/humaneval.jsonl
"""
import json
import subprocess
import sys
import tempfile


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    passed, failed = 0, []
    for r in rows:
        program = f"{r['prompt']}\n\n{r['completion']}\n\n{r['test']}\n\ncheck({r['entry_point']})\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(program)
        try:
            ok = subprocess.run([sys.executable, f.name], capture_output=True, timeout=15).returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
        passed += ok
        if not ok:
            failed.append(r["task_id"])
    print(json.dumps({"n": len(rows), "passed": passed, "pass@1": round(passed / len(rows), 4), "failed": failed}))


if __name__ == "__main__":
    main()
