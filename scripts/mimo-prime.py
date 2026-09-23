#!/usr/bin/env python3
"""Prefix-cache priming for mimo-serve.

Every agent session starts with the same long prefix: system prompt plus tool
definitions (~8.5k tokens for deepseek-harness), ~20 s of prefill per
fresh session. One `max_tokens: 1` request with that exact prefix puts it in
the server's prefix cache, and every later session that starts identically
reuses it (measured: 15.4 s cold -> 0.57 s).

The exact prefix is taken from the agent's OWN captured request: the server
renders the chat template itself, so the cached tokens are exactly what the
agent's next session will produce, up to the first byte that differs.

  mimo-prime analyze [CAPTURE_DIR]            what each agent sends, what varies
  mimo-prime build   [CAPTURE_DIR] [PRIME_DIR] one prime file per agent/prefix
  mimo-prime run     [PRIME_DIR] [--url URL]   send the primes (after a restart)

Captures come from the server with MLX_SERVE_REQUEST_LOG_DIR set.
"""
import hashlib, json, os, sys, time, urllib.request

CAPTURE_DIR = os.path.expanduser("~/.mlx-serve/request-capture")
PRIME_DIR = os.path.expanduser("~/.mlx-serve/prime")
URL = "http://127.0.0.1:11235"
PLACEHOLDER = "."  # the first user turn is replaced: the prefix before it is what is shared


def load(capture_dir):
    recs = []
    for name in sorted(os.listdir(capture_dir)):
        if not name.endswith(".json"):
            continue
        try:
            d = json.load(open(os.path.join(capture_dir, name)))
        except Exception as e:
            print(f"  skip {name}: {e}", file=sys.stderr)
            continue
        if not isinstance(d.get("body"), dict):
            continue
        if (d.get("user_agent") or "").startswith("mimo-prime"):
            continue  # our own priming requests, captured on their way in
        d["_file"] = name
        recs.append(d)
    return recs


def system_text(body, path):
    """The request's system prompt as one string (Anthropic list or OpenAI message)."""
    if path == "/v1/messages":
        s = body.get("system", "")
        if isinstance(s, list):
            return "\n".join(b.get("text", "") for b in s if isinstance(b, dict))
        return s or ""
    for m in body.get("messages", []):
        if m.get("role") in ("system", "developer"):
            c = m.get("content", "")
            return c if isinstance(c, str) else "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def tools_blob(body):
    return json.dumps(body.get("tools") or [], sort_keys=True, ensure_ascii=False)


def agent_key(rec):
    ua = rec.get("user_agent", "") or "?"
    return f"{rec['path']} | {ua.split(' ')[0]}"


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()[:10]


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def analyze(capture_dir):
    recs = load(capture_dir)
    if not recs:
        print(f"no captures in {capture_dir}")
        return
    groups = {}
    for r in recs:
        groups.setdefault(agent_key(r), []).append(r)
    for key, rs in groups.items():
        sys_texts = [system_text(r["body"], r["path"]) for r in rs]
        tools = [tools_blob(r["body"]) for r in rs]
        print(f"\n== {key}: {len(rs)} request(s)")
        print(f"   system: {len(set(map(h, sys_texts)))} distinct of {len(rs)}, {min(map(len, sys_texts))}-{max(map(len, sys_texts))} chars")
        print(f"   tools:  {len(set(map(h, tools)))} distinct, {min(map(len, tools))}-{max(map(len, tools))} chars, "
              f"{len((rs[0]['body'].get('tools') or []))} tools")
        base = sys_texts[0]
        for r, s in zip(rs[1:], sys_texts[1:]):
            i = first_diff(base, s)
            if i >= 0:
                print(f"   system differs from {rs[0]['_file']} at char {i}: "
                      f"{base[max(0, i-40):i+60]!r}\n      vs {s[max(0, i-40):i+60]!r}  ({r['_file']})")
        tb = tools[0]
        for r, t in zip(rs[1:], tools[1:]):
            i = first_diff(tb, t)
            if i >= 0:
                print(f"   tools differ at char {i} ({r['_file']})")


def prime_body(rec):
    """The captured request cut to its shared prefix: system + tools, then a
    placeholder first user turn, max_tokens 1, non-streaming."""
    body = json.loads(json.dumps(rec["body"]))
    msgs = body.get("messages", [])
    keep = []
    for m in msgs:
        if m.get("role") in ("system", "developer"):
            keep.append(m)
            continue
        break
    keep.append({"role": "user", "content": PLACEHOLDER})
    body["messages"] = keep
    body["max_tokens"] = 1
    body["stream"] = False
    for k in ("stream_options", "metadata"):
        body.pop(k, None)
    return body


def build(capture_dir, prime_dir):
    recs = load(capture_dir)
    os.makedirs(prime_dir, exist_ok=True)
    os.chmod(prime_dir, 0o700)
    seen = {}
    for r in recs:
        body = prime_body(r)
        sig = h(r["path"] + system_text(r["body"], r["path"]) + tools_blob(r["body"]) + body.get("model", ""))
        seen.setdefault(sig, (r, body))
    for sig, (r, body) in seen.items():
        name = f"{agent_key(r).split('|')[1].strip().replace('/', '_') or 'agent'}-{sig}.json"
        with open(os.path.join(prime_dir, name), "w") as f:
            json.dump({"path": r["path"], "source": r["_file"], "body": body}, f, ensure_ascii=False)
        print(f"wrote {name}  ({r['path']}, system {len(system_text(r['body'], r['path']))} chars, "
              f"{len(r['body'].get('tools') or [])} tools, from {r['_file']})")


def run(prime_dir, url):
    names = sorted(n for n in os.listdir(prime_dir) if n.endswith(".json")) if os.path.isdir(prime_dir) else []
    if not names:
        print(f"no prime files in {prime_dir}")
        return
    for n in names:
        d = json.load(open(os.path.join(prime_dir, n)))
        req = urllib.request.Request(url + d["path"], data=json.dumps(d["body"]).encode(),
                                     headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01",
                                              "x-api-key": "local", "User-Agent": "mimo-prime/1"})
        t = time.time()
        try:
            r = json.load(urllib.request.urlopen(req, timeout=1800))
            u = r.get("usage", {})
            p = u.get("prompt_tokens", u.get("input_tokens", "?"))
            c = u.get("prompt_tokens_details", {}).get("cached_tokens", u.get("cache_read_input_tokens", 0))
            print(f"primed {n}: {p} prompt tokens ({c} already cached) in {time.time()-t:.1f}s", flush=True)
        except Exception as e:
            print(f"FAILED {n}: {e}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("analyze", "build", "run"):
        print(__doc__)
        sys.exit(1)
    if a[0] == "analyze":
        analyze(a[1] if len(a) > 1 else CAPTURE_DIR)
    elif a[0] == "build":
        build(a[1] if len(a) > 1 else CAPTURE_DIR, a[2] if len(a) > 2 else PRIME_DIR)
    else:
        url = URL
        if "--url" in a:
            url = a[a.index("--url") + 1]
            a = [x for i, x in enumerate(a) if x != "--url" and (i == 0 or a[i - 1] != "--url")]
        run(a[1] if len(a) > 1 else PRIME_DIR, url)
