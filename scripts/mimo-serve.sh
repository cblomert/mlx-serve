#!/usr/bin/env bash
# Serve MiMo-V2.6-Flash-RL on the LAN via mlx-serve.
#
# This runs a LOCAL FORK BUILD, not Homebrew's mlx-serve and no longer mlx-lm:
# upstream has no `mimo_v2_flash` arm (checked against main at v26.9.5). The
# fork adds one -- config arm, attention arm, weight binding, and a fix to the
# quantization solver for this checkpoint's MIXED affine/mxfp4 banks.
#
# What the switch off mlx-lm buys: continuous batching, the hot + disk prefix
# cache, the Anthropic /v1/messages surface, and /metrics. mlx-lm had none of
# those and no reasoning_content field either.
#
# Source of the build (the binary is useless without it):
#   git clone ~/.local/opt/mlx-serve-mimo/src-snapshot/mlx-serve-mimo.bundle
#   git checkout feat/mimo-v2-flash
#
# --- modalities (all native in the fork; all verified through this server) --
#
#   text + reasoning   /v1/chat/completions, /v1/messages, /api/chat
#   images             image_url parts (JPEG/PNG/WebP). MiMo ViT, 28 blocks,
#                      CLIP-normalised, area capped at 1536^2 px.
#   video              video_url {"frames":[...data URLs...], "fps":2}; frames
#                      are grouped 2 per temporal patch and framed with MiMo's
#                      "MM:SS" timestamps. Default 2 fps.
#   audio IN           input_audio {"data": base64 WAV, "format":"wav"} (any
#                      rate/channels) or raw 16 kHz f32 PCM. MiMo audio
#                      tokenizer + encoder, 6.25 tokens per second of audio;
#                      >60 s clips are segmented like the reference.
#   audio OUT          POST /v1/audio/speech {"model":"ddalcu/Kokoro-82M-MLX-Serve",
#                      "input":"...","voice":"af_heart"} -> 24 kHz WAV. The
#                      MiMo checkpoint has NO speech-generation head (the
#                      tokenizer decoder ships, but nothing emits audio codes),
#                      so speech out is Kokoro, served beside MiMo from
#                      MIMO_TTS_DIR. Unset MIMO_TTS_DIR= to drop it.
#   MTP                --mtp is available but OFF: correct, but slower on this
#                      sparse MoE (verifying W tokens pulls ~W x the experts).
#
# The encoders add ~3 GB resident (vision 1.4, audio 1.3, Kokoro 0.3) on top
# of the 164.4 GB trunk -- inside the headroom figures below.
set -euo pipefail

MODEL="$HOME/.mlx-serve/models/Vontra/MiMo-V2.6-Flash-RL-MLX-4bit-MTP"
MLX_SERVE_HOME="${MLX_SERVE_HOME:-$HOME/.local/opt/mlx-serve-mimo}"
BIN="$MLX_SERVE_HOME/zig-out/bin/mlx-serve"

MIMO_HOST="${MIMO_HOST:-0.0.0.0}"
MIMO_PORT="${MIMO_PORT:-11235}"          # 11234 belongs to qwen-serve

# --- memory, which is the ONLY thing that sizes this server -----------------
#
# Weights are 164.4 GB resident (measured, matching the publisher). KV is
# 22.5 KB/token: 9 full-attention layers x (4 kv heads x 192 K + 4 x 128 V) x
# 2 B. The 39 sliding layers hold only their 128-token window -- ~26 MB total,
# and they do NOT scale with context. Against a 240 GB wired limit:
#
#   streams x window      KV      total    headroom
#   1 x 1M              24.2 GB  188.6 GB  51.4 GB
#   2 x 1M              48.3 GB  212.7 GB  27.3 GB   <- default
#   4 x 512k            48.3 GB  212.7 GB  27.3 GB
#   1 x 1M + 3 x 512k   60.4 GB  224.8 GB  15.2 GB   marginal
#   2 x 1M + 2 x 512k   72.5 GB  236.9 GB   3.1 GB   DO NOT -- prefill compute
#                                                    buffers do not fit in 3 GB
#   4 x 1M              96.6 GB  261.1 GB     over
#
# --ctx-size is a GLOBAL cap, not per stream: there is no way to grant one
# agent 1M and another 512k. So the safe pairing is (concurrency x ctx) whose
# WORST case fits. Two at the full native window is the default; for four
# agents set MIMO_CONCURRENT=4 MIMO_CTX=524288, which costs the same KV.
#
# The prefix cache is NOT free headroom: the server clamps it to
#   ceiling - (weights + max_concurrent x ctx x 22.5 KB + prefill transients)
# and at 2 x 1M that residual is ZERO -- measured, the cache came up at
# "mem-cap=0.0 MB" and refused every entry. The server's own clamp is the same
# arithmetic as the table above, arrived at independently, which is a good sign
# the table is right.
#
# Three agents at 512k is 36.2 GB of REAL KV -> 200.6 GB, 39.4 GB of headroom.
# The planner's inflated row bytes will still clamp the cache's BYTE budget to
# zero, which is not the same as disabling it: the count cap
# (--prefix-cache-entries) still holds entries, and warm reuse was measured at
# 6277 of 6294 tokens, 18.6s -> 1.7s. So the clamp costs an eviction policy,
# not the cache.
#
# For one agent at the full native window instead: MIMO_CONCURRENT=1
# MIMO_CTX=1048576 (24.2 GB, the same real cost as 2 x 512k).
# 3 streams at a 1M CEILING: the intended shape is one deep agent plus two
# shallower ones (1x1M + 2x512k = 48.3 GB KV -> 212.7 GB, 27.3 GB headroom).
#
# READ THIS BEFORE RAISING EITHER NUMBER. --ctx-size is a GLOBAL ceiling, not a
# per-stream grant: nothing here stops all three streams growing to 1M, and that
# worst case is 72.5 GB KV -> 236.9 GB against a 240 GB wired limit. 3.1 GB does
# not cover prefill compute buffers, so it OOMs -- a hard process kill, not a
# 400. The split is enforced by the CLIENTS capping their own context; this
# server cannot do it for you.
#
# If you would rather have the guarantee than the deep window, set
# MIMO_CTX=524288: 3 x 512k is 36.2 GB and fits even if every stream maxes out.
MIMO_CONCURRENT="${MIMO_CONCURRENT:-3}"
MIMO_CTX="${MIMO_CTX:-1048576}"

MIMO_PREFIX_MEM="${MIMO_PREFIX_MEM:-20GB}"

# Prefill chunk CEILING. MiMo's 192-wide heads have no fused SDPA kernel, so
# attention composes [heads x chunk x keys] scores -- on the 39 sliding layers
# that is chunk x (chunk + 128), i.e. it grows with chunk^2. Measured at a
# 12.5k prompt: 1024 -> 393, 2048 -> 400, 4096 -> 341, 8192 -> 250 tok/s. The
# server's own score budget still shrinks the chunk further at long context
# (512 from ~17k tokens), which is also the measured optimum there (67k: 512
# -> 265 tok/s, 2048 -> 237). Without this flag the load-time sizer picks 4096.
MIMO_PREFILL_CHUNK="${MIMO_PREFILL_CHUNK:-2048}"

# Prefix-cache priming. Every agent session starts with the same system prompt
# + tool definitions (~10k tokens, ~25 s of prefill cold). After the server is
# healthy, `mimo-prime run` replays each prime file in MIMO_PRIME_DIR (one per
# agent prefix, built from captured requests by `mimo-prime build`) with
# max_tokens 1, so the first session after a restart starts warm. Measured:
# 19 s -> 0.9 s for a fresh session. MIMO_PRIME_DIR= disables it.
MIMO_PRIME_DIR="${MIMO_PRIME_DIR-$HOME/.mlx-serve/prime}"
# Request capture for (re)building the primes: MIMO_REQUEST_LOG=1 writes every
# chat request body to ~/.mlx-serve/request-capture (prompts and code included
# -- a private dir; turn it off again once the capture is done).
MIMO_REQUEST_LOG="${MIMO_REQUEST_LOG:-0}"
if [ "$MIMO_REQUEST_LOG" = "1" ]; then
  export MLX_SERVE_REQUEST_LOG_DIR="$HOME/.mlx-serve/request-capture"
  mkdir -p "$MLX_SERVE_REQUEST_LOG_DIR" && chmod 700 "$MLX_SERVE_REQUEST_LOG_DIR"
fi
MIMO_TTS_DIR="${MIMO_TTS_DIR-$HOME/.mlx-serve/tts-models}"   # Kokoro lives here
MIMO_PREFIX_DISK="${MIMO_PREFIX_DISK:-48GB}"

if [ ! -x "$BIN" ]; then
  echo "mlx-serve fork build not found at:" >&2
  echo "  $BIN" >&2
  echo >&2
  echo "Upstream mlx-serve cannot load this model; rebuild the fork from" >&2
  echo "  $MLX_SERVE_HOME/src-snapshot/mlx-serve-mimo.bundle" >&2
  exit 1
fi

if [ ! -d "$MODEL" ]; then
  echo "Model not found at: $MODEL" >&2
  echo "  hf download Vontra/MiMo-V2.6-Flash-RL-MLX-4bit-MTP \\" >&2
  echo "     --local-dir \"$MODEL\" --max-workers 16      # ~160 GB" >&2
  exit 1
fi

# Hard guard: MiMo is 164.4 GB and qwen-serve holds ~68 GB. Together that is
# 232 GB before a single KV byte, against a 240 GB wired limit. Starting both
# does not degrade, it OOMs.
if pgrep -f "mlx-serve-yarn" >/dev/null 2>&1; then
  echo "qwen-serve is running and there is not enough memory for both." >&2
  echo "  MiMo 164.4 GB + Qwen ~68 GB = 232 GB before any KV," >&2
  echo "  against iogpu.wired_limit_mb = $(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo '?') MB." >&2
  echo >&2
  echo "  launchctl bootout gui/\$(id -u)/com.zerodebug.qwen-serve" >&2
  exit 1
fi

# Thinking is ON by default for this arch (the fork adds it to
# defaultEnableThinking): MiMo's template commits `<think></think>` ONLY when
# enable_thinking is false, so defaulting off would spend two tokens
# SUPPRESSING the pass the checkpoint was RL-tuned to run. Per request,
# `reasoning_effort: "none"`, `enable_thinking: false` (top level or inside
# vLLM-style `chat_template_kwargs`) or `thinking: {"type":"disabled"}` turns it off.
#
# Reasoning is served on every surface: `reasoning_content` on
# /v1/chat/completions (streaming too, as `delta.reasoning_content`), a
# `thinking` content block on /v1/messages (`thinking_delta` when streaming),
# and `message.thinking` on /api/chat. All five verified.
#
# max_tokens covers REASONING + ANSWER, and reasoning goes first. Measured: a
# 400-token cap produced 1727 chars of thinking and ZERO content. Clients
# should budget >= 200 for anything expecting a real reply; MIMO_REASONING_BUDGET
# is the server-side backstop that caps thinking per request so a tight client
# cap cannot yield an empty answer. Unset = unlimited, which is the shipped
# behaviour -- set it only if you cannot fix the caller.
MIMO_REASONING_BUDGET="${MIMO_REASONING_BUDGET:-0}"

CRASHLOG="${MIMO_CRASHLOG:-$HOME/.mlx-serve/logs/mimo-serve-stderr.log}"
mkdir -p "$(dirname "$CRASHLOG")"
{
  echo
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') start: host=$MIMO_HOST port=$MIMO_PORT" \
       "concurrent=$MIMO_CONCURRENT ctx=$MIMO_CTX cache=$MIMO_PREFIX_MEM ==="
} >> "$CRASHLOG"

TTS_ARGS=()
if [ -n "$MIMO_TTS_DIR" ] && [ -d "$MIMO_TTS_DIR" ]; then
  TTS_ARGS+=(--model-dir "$MIMO_TTS_DIR")
fi

if [ -n "$MIMO_PRIME_DIR" ] && [ -x "$HOME/bin/mimo-prime" ] && ls "$MIMO_PRIME_DIR"/*.json >/dev/null 2>&1; then
  mkdir -p "$HOME/.mlx-serve/logs"
  (
    for _ in $(seq 1 300); do
      curl -sf "http://127.0.0.1:$MIMO_PORT/health" >/dev/null 2>&1 && break
      sleep 2
    done
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') priming from $MIMO_PRIME_DIR"
    "$HOME/bin/mimo-prime" run "$MIMO_PRIME_DIR" --url "http://127.0.0.1:$MIMO_PORT"
  ) >> "$HOME/.mlx-serve/logs/mimo-prime.log" 2>&1 < /dev/null &
fi

REASON_ARGS=()
if [ "$MIMO_REASONING_BUDGET" != "0" ]; then
  REASON_ARGS+=(--reasoning-budget "$MIMO_REASONING_BUDGET")
fi

exec "$BIN" \
  --model "$MODEL" \
  --serve \
  --host "$MIMO_HOST" \
  --port "$MIMO_PORT" \
  --ctx-size "$MIMO_CTX" \
  --max-concurrent "$MIMO_CONCURRENT" \
  --prefix-cache-mem "$MIMO_PREFIX_MEM" \
  --prefix-cache-entries 32 \
  --prefix-cache-disk "$MIMO_PREFIX_DISK" \
  --tokenize-cache-entries 8 \
  --prefill-chunk "$MIMO_PREFILL_CHUNK" \
  ${TTS_ARGS[@]+"${TTS_ARGS[@]}"} \
  ${REASON_ARGS[@]+"${REASON_ARGS[@]}"} \
  --metrics \
  --timeout 1800 \
  2> >(tee -a "$CRASHLOG" >&2)
