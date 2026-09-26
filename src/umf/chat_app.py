"""Local side-by-side chat with Tinker checkpoints.

Serves a small web page on localhost with two chat panes. One message box sends
to both panes; each pane has its own model and its own history, so the same
conversation can be compared across checkpoints. The Tinker key stays in this
process and is never sent to the browser.

The model list is, for each base model used in ``--runs``, the raw base model
and its warm-up adapter, then every sampler checkpoint in each ``--runs`` log
directory (re-read on every page
load, so checkpoints appear while a run is still training).

    python -m umf.chat_app --runs logs/warmup_utility/warm logs/warmup_utility/cold
    # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from umf.chat_format import RENDERER_NAME

# Reference models offered for every base model that appears in --runs.
WARMUPS = {
    "Qwen/Qwen3.6-35B-A3B": "tinker://6128d6e1-c39f-5dbf-ba00-4571f6a55571:train:0/sampler_weights/final",
    "Qwen/Qwen3-8B": "tinker://3d025738-faaf-5a1a-8ec4-635eca1717ca:train:0/sampler_weights/final",
}
BASE_PREFIX = "base:"  # model id of a raw base model, e.g. "base:Qwen/Qwen3-8B"


def run_model_name(run_dir: Path) -> str | None:
    """Base model of a training run, from the config the cookbook writes."""
    path = run_dir / "config.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("model_name")


def run_checkpoints(run_dir: Path, steps_per_epoch: int | None) -> list[dict]:
    """Sampler checkpoints of one training run, in training order."""
    path = run_dir / "checkpoints.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("sampler_path"):
            continue
        # The name is the global step ("000150"); "batch" restarts every epoch.
        step = int(r["name"]) if r["name"].isdigit() else None
        label = f"{run_dir.name} · {r['name']}"
        if r["name"] == "final":
            label = f"{run_dir.name} · final (end of epoch {r.get('epoch')})"
        elif step is not None:
            label = f"{run_dir.name} · step {step}"
            if steps_per_epoch:
                label += f" (epoch {step / steps_per_epoch:g})"
        out.append({"id": r["sampler_path"], "label": label})
    return out


class Backend:
    def __init__(self, run_dirs: list[Path], steps_per_epoch: int | None):
        self.run_dirs = run_dirs
        self.steps_per_epoch = steps_per_epoch
        self.service = tinker.ServiceClient()
        self._clients: dict[str, tinker.SamplingClient] = {}
        self._renderers: dict[str, tuple] = {}
        self._model_of: dict[str, str] = {}  # model id -> base model name
        self._lock = threading.Lock()

    def models(self) -> list[dict]:
        runs = [(d, run_model_name(d)) for d in self.run_dirs]
        names = sorted({n for _, n in runs if n}, key=lambda n: list(WARMUPS).index(n) if n in WARMUPS else 99)
        groups = []
        for n in names:
            refs = [{"id": BASE_PREFIX + n, "label": f"Base {n.split('/')[-1]} (no adapter)"}]
            if n in WARMUPS:
                refs.append({"id": WARMUPS[n], "label": f"{n.split('/')[-1]} warm-up adapter"})
            groups.append({"group": f"Reference · {n.split('/')[-1]}", "models": refs})
            for m in refs:
                self._model_of[m["id"]] = n
        for d, n in runs:
            ckpts = run_checkpoints(d, self.steps_per_epoch)
            for m in ckpts:
                self._model_of[m["id"]] = n
            short = n.split("/")[-1] if n else "?"
            groups.append({"group": f"{d.name} ({short})", "models": ckpts})
        return groups

    def _model_name(self, model_id: str) -> str:
        if model_id.startswith(BASE_PREFIX):
            return model_id[len(BASE_PREFIX):]
        if model_id not in self._model_of:
            self.models()
        return self._model_of[model_id]

    def _renderer(self, model_name: str):
        with self._lock:
            if model_name not in self._renderers:
                tok = get_tokenizer(model_name)
                self._renderers[model_name] = (tok, renderers.get_renderer(RENDERER_NAME, tokenizer=tok))
            return self._renderers[model_name]

    def _client(self, model_id: str) -> tinker.SamplingClient:
        with self._lock:
            if model_id not in self._clients:
                if model_id.startswith(BASE_PREFIX):
                    c = self.service.create_sampling_client(base_model=model_id[len(BASE_PREFIX):])
                else:
                    c = self.service.create_sampling_client(model_path=model_id)
                self._clients[model_id] = c
            return self._clients[model_id]

    def chat(self, model_id: str, messages: list[dict], temperature: float, max_tokens: int) -> dict:
        tokenizer, renderer = self._renderer(self._model_name(model_id))
        convo = [renderers.Message(role=m["role"], content=m["content"]) for m in messages]
        prompt = renderer.build_generation_prompt(convo)
        params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        )
        result = self._client(model_id).sample(prompt, num_samples=1, sampling_params=params).result()
        tokens = result.sequences[0].tokens
        message, _ = renderer.parse_response(tokens)
        return {
            "text": message["content"],
            "raw": tokenizer.decode(tokens),
            "hit_cap": len(tokens) >= max_tokens,
            "n_tokens": len(tokens),
        }


def make_handler(backend: Backend):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            if self.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/models":
                self._json(200, backend.models())
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            if self.path != "/chat":
                self._send(404, b"not found", "text/plain")
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                out = backend.chat(
                    req["model"],
                    req["messages"],
                    float(req.get("temperature", 1.0)),
                    int(req.get("max_tokens", 1024)),
                )
                self._json(200, out)
            except Exception as e:  # noqa: BLE001 - report to the page
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, format, *args):  # noqa: A002 - keep the terminal quiet
            pass

    return Handler


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Checkpoint Chat</title>
<style>
:root {
  --bg: #f7f7f5; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6b66; --border: #dddcd6;
  --user: #fbeee6; --user-border: #e3a27c; --asst: #f1f1ee; --accent: #d0591b; --warn: #b3142a;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #161615; --panel: #1f1f1d; --text: #ecebe6; --muted: #9a998f; --border: #34332f;
    --user: #3a261b; --user-border: #9c5530; --asst: #2a2a27; --accent: #ec7a3c; --warn: #f06b7d;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  height: 100vh; display: flex; flex-direction: column; }
header { padding: 10px 16px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
  border-bottom: 1px solid var(--border); }
header h1 { font-size: 15px; margin: 0; font-weight: 600; }
header label { color: var(--muted); font-size: 13px; display: flex; gap: 6px; align-items: center; }
header input[type=number] { width: 70px; }
main { flex: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 12px 16px; min-height: 0; }
@media (max-width: 800px) { main { grid-template-columns: 1fr; } }
.pane { background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  display: flex; flex-direction: column; min-height: 0; }
.pane-head { padding: 8px 10px; border-bottom: 1px solid var(--border); display: flex; gap: 8px; align-items: center; }
.pane-head select { flex: 1; min-width: 0; }
.log { flex: 1; overflow-y: auto; padding: 10px; display: flex; flex-direction: column; gap: 8px; }
.msg { padding: 8px 10px; border-radius: 6px; white-space: pre-wrap; word-wrap: break-word; }
.msg.user { background: var(--user); border: 1px solid var(--user-border); align-self: flex-end; max-width: 90%; }
.msg.assistant { background: var(--asst); }
.msg .meta { font-size: 11px; color: var(--muted); margin-top: 4px; }
.msg .meta.warn { color: var(--warn); }
.msg details { margin-top: 4px; font-size: 12px; }
.msg details pre { white-space: pre-wrap; font-size: 11px; color: var(--muted); margin: 4px 0 0; }
.msg.error { color: var(--warn); }
footer { padding: 10px 16px 14px; display: flex; gap: 8px; border-top: 1px solid var(--border); }
textarea { flex: 1; resize: vertical; min-height: 44px; font: inherit; padding: 8px;
  background: var(--panel); color: var(--text); border: 1px solid var(--border); border-radius: 6px; }
button, select, input { font: inherit; color: var(--text); background: var(--panel);
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; }
button.primary { background: var(--accent); color: #fff; border-color: var(--accent); padding: 6px 16px; }
button:disabled { opacity: .5; }
</style>
</head>
<body>
<header>
  <h1>Checkpoint chat</h1>
  <label>Temperature <input id="temp" type="number" min="0" max="2" step="0.1" value="1.0"></label>
  <label>Max tokens <input id="maxtok" type="number" min="16" max="8192" step="64" value="1024"></label>
  <label><input id="both" type="checkbox" checked> Send to both panes</label>
  <button id="reload">Refresh model list</button>
</header>
<main id="panes"></main>
<footer>
  <textarea id="input" placeholder="Message (Enter to send, Shift+Enter for a new line)"></textarea>
  <button class="primary" id="send">Send</button>
</footer>
<script>
const panes = [];
let groups = [];

function makePane(i, defaultId) {
  const el = document.createElement("section");
  el.className = "pane";
  el.innerHTML = `<div class="pane-head"><select></select><button>Clear</button></div><div class="log"></div>`;
  const pane = { el, select: el.querySelector("select"), log: el.querySelector(".log"), history: [], defaultId };
  el.querySelector("button").onclick = () => { pane.history = []; pane.log.innerHTML = ""; };
  pane.select.onchange = () => { pane.history = []; pane.log.innerHTML = ""; };
  document.getElementById("panes").appendChild(el);
  panes.push(pane);
}

function fillSelect(pane) {
  const keep = pane.select.value || pane.defaultId;
  pane.select.innerHTML = "";
  for (const g of groups) {
    const og = document.createElement("optgroup");
    og.label = g.group;
    for (const m of g.models) {
      const o = document.createElement("option");
      o.value = m.id; o.textContent = m.label;
      og.appendChild(o);
    }
    pane.select.appendChild(og);
  }
  if ([...pane.select.options].some(o => o.value === keep)) pane.select.value = keep;
  else if (pane.select.options.length) pane.select.selectedIndex = Math.min(panes.indexOf(pane), pane.select.options.length - 1);
}

async function loadModels() {
  groups = await (await fetch("/models")).json();
  panes.forEach(fillSelect);
}

function addMsg(pane, role, text, extra) {
  const d = document.createElement("div");
  d.className = "msg " + role;
  d.textContent = text;
  if (extra) d.appendChild(extra);
  pane.log.appendChild(d);
  pane.log.scrollTop = pane.log.scrollHeight;
  return d;
}

async function sendTo(pane, text) {
  pane.history.push({ role: "user", content: text });
  addMsg(pane, "user", text);
  const pending = addMsg(pane, "assistant", "…");
  try {
    const r = await fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: pane.select.value, messages: pane.history,
        temperature: parseFloat(document.getElementById("temp").value),
        max_tokens: parseInt(document.getElementById("maxtok").value) }) });
    const out = await r.json();
    if (!r.ok) throw new Error(out.error || r.statusText);
    pane.history.push({ role: "assistant", content: out.text });
    pending.textContent = out.text;
    const meta = document.createElement("div");
    meta.className = "meta" + (out.hit_cap ? " warn" : "");
    meta.textContent = `${out.n_tokens} tokens` + (out.hit_cap ? " · hit the token cap (never ended its turn)" : "");
    const raw = document.createElement("details");
    raw.innerHTML = "<summary>Raw sampled tokens</summary><pre></pre>";
    raw.querySelector("pre").textContent = out.raw;
    pending.append(meta, raw);
  } catch (e) {
    pending.className = "msg error";
    pending.textContent = "Error: " + e.message;
    pane.history.pop();
  }
}

async function send() {
  const box = document.getElementById("input");
  const text = box.value.trim();
  if (!text) return;
  box.value = "";
  const btn = document.getElementById("send");
  btn.disabled = true;
  const targets = document.getElementById("both").checked ? panes : [panes[0]];
  await Promise.all(targets.map(p => sendTo(p, text)));
  btn.disabled = false;
  box.focus();
}

document.getElementById("send").onclick = send;
document.getElementById("input").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
document.getElementById("reload").onclick = loadModels;
makePane(0, null);
makePane(1, null);
loadModels();
</script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", default=[], help="training log dirs with checkpoints.jsonl")
    p.add_argument("--steps-per-epoch", type=int, default=None, help="label checkpoints with epochs")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    backend = Backend([Path(r) for r in args.runs], args.steps_per_epoch)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(backend))
    print(f"chat at http://127.0.0.1:{args.port}  (Ctrl+C to stop)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
