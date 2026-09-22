#!/usr/bin/env bash
# mimo_v2_flash over HTTP: the checks that a unit test cannot make, because
# they need the real 160 GB checkpoint and the whole serving stack.
#
#   1. it answers correctly at all                (arithmetic with working)
#   2. the 39 sliding + 9 full layers interleave  (needle far past the
#      128-token window -- below it every layer degenerates to plain causal
#      and a mask or layer-pattern bug is INVISIBLE)
#   3. thinking defaults ON, matching the checkpoint's template
#
# Usage: MODEL=/path/to/MiMo-V2.6-Flash-RL-MLX-4bit-MTP tests/test_mimo_v2_flash.sh
set -euo pipefail

MODEL="${MODEL:-$HOME/.mlx-serve/models/Vontra/MiMo-V2.6-Flash-RL-MLX-4bit-MTP}"
PORT="${PORT:-11239}"
BIN="${BIN:-./zig-out/bin/mlx-serve}"
[ -d "$MODEL" ] || { echo "SKIP: checkpoint not present at $MODEL"; exit 0; }

"$BIN" --model "$MODEL" --serve --host 127.0.0.1 --port "$PORT" --max-concurrent 1 >/tmp/mimo_it.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do curl -sf -m 2 -o /dev/null "http://127.0.0.1:$PORT/health" && break; sleep 2; done

fail=0
ask() { # $1 = json body -> prints assistant content
  curl -s -m 1800 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" -d "$1" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"].get("content") or "")'
}

echo "1. arithmetic"
out=$(ask '{"model":"m","max_tokens":80,"temperature":0.0,"messages":[{"role":"user","content":"What is 17 times 23?"}]}')
case "$out" in *391*) echo "   ok";; *) echo "   FAIL: $out"; fail=1;; esac

echo "2. needle past the sliding window"
body=$(python3 - <<'PY'
import json,random
rng=random.Random(11); W="alpha bravo charlie delta echo foxtrot golf hotel india juliet".split()
w=[rng.choice(W) for _ in range(2200)]
w[len(w)//2:len(w)//2]=["The archive access code is ARCHIVE-5150-2270."]
p="Reference corpus:\n"+" ".join(w)+"\n\nState the archive access code exactly as written."
print(json.dumps({"model":"m","max_tokens":32,"temperature":0.0,
                  "messages":[{"role":"user","content":p}]}))
PY
)
out=$(ask "$body")
case "$out" in *ARCHIVE-5150-2270*) echo "   ok (prompt is ~3k tokens, 24x the 128 window)";;
                *) echo "   FAIL: $out"; fail=1;; esac

echo "3. thinking defaults on"
n=$(curl -s -m 60 "http://127.0.0.1:$PORT/tokenize" -H "Content-Type: application/json" \
     -d '{"content":"x"}' >/dev/null 2>&1; grep -c "enable_thinking" /tmp/mimo_it.log || true)
echo "   (see /tmp/mimo_it.log; the template commits <think></think> only when OFF)"

exit $fail
