# Local and open-source LLMs

Yes, local models are supported. Anything that serves the OpenAI Chat Completions API with
tools works: llama.cpp's `llama-server`, vLLM, Ollama, LM Studio, and any other
OpenAI-compatible endpoint. No API key is needed.

```bash
uvx --from "quackd[openai]" quackd run find-and-kick --provider ollama --model qwen3:8b
uvx --from "quackd[openai]" quackd run find-and-kick --provider vllm --model Qwen/Qwen3-8B
uvx --from "quackd[openai]" quackd run find-and-kick --provider llamacpp
uvx --from "quackd[openai]" quackd run find-and-kick --provider lmstudio
uvx --from "quackd[openai]" quackd run find-and-kick --provider local --base-url http://gpu-box:8000/v1
```

The `openai` extra is the `openai` Python package, which is the client for all of these.
Leave `--model` off and quackd asks the server for its model list and takes the first one.
`--model` here is free text: any id your server serves is accepted, because the model catalogue
`quackd list-models` prints, and that `--model` is checked against, covers the cloud vendors
only. A local preset is never refused for naming something the catalogue has not heard of.

| `--provider` | Default address | Override |
|---|---|---|
| `ollama` | `http://localhost:11434/v1` | `--base-url` or `QUACKD_BASE_URL` |
| `vllm` | `http://localhost:8000/v1` | same |
| `llamacpp` | `http://localhost:8080/v1` | same |
| `lmstudio` | `http://localhost:1234/v1` | same |
| `local` | none, you must pass one | same |

`quackd doctor` probes all four default addresses and prints which servers are up and what
they serve.

## Server setup

Tool calling has to be switched on in some servers. These are the flags that matter.

**Ollama**

```bash
ollama pull qwen3:8b          # any model whose card says it supports tools
ollama serve                  # usually already running as a service
quackd run find-and-kick --provider ollama --model qwen3:8b
```

**llama.cpp**

```bash
llama-server -m model.gguf --jinja --port 8080     # --jinja enables the tool-calling chat templates
quackd run find-and-kick --provider llamacpp
```

**vLLM**

```bash
vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes
quackd run find-and-kick --provider vllm --model Qwen/Qwen3-8B
```

The `--tool-call-parser` value depends on the model family (`hermes` for Qwen and Hermes
models, `llama3_json` for Llama 3.x, `mistral` for Mistral). vLLM's docs list the pairs.

**LM Studio**

Developer tab → Start Server (default port 1234), load a model that supports tools, then
`quackd run find-and-kick --provider lmstudio`.

## What to expect from small models

quackd asks for exactly one tool call per turn. Frontier models do this reliably. Small
local models sometimes answer with JSON in plain text instead of a native tool call, or
call a verb that is not allowed, or add chatter. Three things make that workable:

1. **Text fallback.** If a reply has no native tool call, quackd looks for a JSON object
   like `{"name": "walk_to", "arguments": {"target": "ball"}}` in the text and uses it. Only
   the answer is read: an inline `<think>...</think>` block is split off first, so a verb the
   model weighed inside its reasoning and dropped is never executed. The transcript marks a
   rescued turn with `stop_reason: "text_fallback"` so you can see how often it happened. The system prompt tells local models this shape exists.
2. **One retry.** A turn with no usable call is re-prompted once, then counts as a failure.
   Budgets still apply.
3. **The executor never trusts the model.** A disallowed verb or bad parameters come back
   as feedback, not as robot motion.

Vision is off by default for local providers because most local models are text only and
servers reject image parts. The text observation already carries what the camera detected
(`ball at bearing 18° left, ~0.6 m`), which is the designed path. For a vision model
(qwen2.5-vl, gemma3, llava and friends) pass `--vision` or set `QUACKD_VISION=1`.

## Knobs

| Setting | Values | Default for local |
|---|---|---|
| `--model` / `QUACKD_MODEL` | any id the server serves, checked against no catalogue | first entry of `/v1/models` |
| `--base-url` / `QUACKD_BASE_URL` | `http://host:port/v1` | the preset's address |
| `--api-key` / `LOCAL_API_KEY` | any string | `not-needed` (servers ignore it) |
| `QUACKD_TOOL_CHOICE` | `auto`, `required`, `none` | `auto` (`none` omits the field for servers that reject it) |
| `--vision` / `QUACKD_VISION` | on, off | off |

`parallel_tool_calls` is never sent to local servers, because some reject unknown fields.

Add physics by asking for both extras and naming the backend:

```bash
uvx --from "quackd[openai,mujoco]" quackd run find-and-kick --provider ollama --model qwen3:8b --robot microduck:mujoco
```

The duck then walks on upstream's own trained policy instead of sliding around a cartoon. It
also undershoots what it is asked for, which is a harder task for a small model and which the
run states in `report_state` ([ADR-0030](adr/0030-mujoco-physics-backend.md)).

## The same duck in a browser, with no quackd installed

[`web/`](../web/README.md) is a static page that runs the physics simulator through MuJoCo's
WebAssembly build and drives it from any OpenAI compatible server, so a local model can pilot
the duck with no key and no `pip install`. It is live at <https://www.quackd.org/simulator>, and
that copy asks nothing of you first: same page, same Local option, same base-URL box, no checkout.

If your model server is on the same machine, a checkout is still the more reliable route, because
that live copy is served over https and your server is not. Browsers disagree about that pairing:
`http://localhost` counts as a trustworthy origin, so some of them let an https page call a
plaintext server on your own machine and others refuse it as mixed content or want a permission
first. A copy you serve yourself is plain http at both ends, so there is no such argument to have,
and it is also what you run when you are changing the page. The page wants a server in front of it, for two
reasons: browsers refuse ES modules over `file://`, and the page expects to be mounted at
`/simulator`, so every local reference in it is absolute. `web/serve.py` is that server —
stdlib only, so it is Python but not quackd:

```bash
python web/serve.py            # then open http://localhost:8000/simulator/
```

`python -m http.server --directory web` will not do: it serves the HTML and then 404s
`/simulator/style.css` and `/simulator/src/app.js`, because nothing answers on the mount at
the root. If port 8000 is already a vLLM, which the table above assumes it is, then
`python web/serve.py 8001` moves the page rather than the model server.

Either copy is driven the same way. Pick Local in the page and give it your base URL. Ollama has
to be told to accept the page (`OLLAMA_ORIGINS=* ollama serve`), and llama.cpp, vLLM and LM
Studio need the same CORS permission, whichever copy you opened. The keyboard beside the sentence
box is live at the same time as the model, so you can take the duck off a stalled local model
mid-run with `W` and the transcript records the handover.
What the page has and has not been run against is in [web/README.md](../web/README.md).

## Honest notes

- Which local model pilots the duck well is an open question we have not measured. The
  loop was designed so that a weak planner degrades the task, never the robot's balance.
  There is exactly one data point, and it is not ours: the contributor who built memory
  between runs ran `find-and-kick` against **Qwen 2.5 Coder 14B on LM Studio**, seeds 5 and
  6, both successes with memory read and written, and reported that the model ignored a
  memory hint sitting in the system prompt but followed the same instruction once it was a
  numbered step in the `.duck` body. That is one model, one machine, two seeds, and no
  transcript in this repository. If you run one, please share the transcript in a
  Discussion: it is the cheapest way to make this section shorter.
- The cloud providers keep their stricter settings (`tool_choice="required"`,
  `parallel_tool_calls=False`). Only the local presets use the relaxed ones.
- Ollama, vLLM, llama.cpp and LM Studio evolve quickly. If a flag above is stale, open an
  issue with the server version.
