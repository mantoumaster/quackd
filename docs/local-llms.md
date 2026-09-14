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

Qwen3 thinks before it answers unless the request says otherwise, and the switch is a chat
template argument rather than a sampling parameter. One reported step of `find-and-kick` spent
150 s and 1717 output tokens on the reasoning before deciding (#12). There are two places to
turn it off. On a server you run yourself, do it once at serve time:

```bash
vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes \n  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'
```

On a server somebody else runs, or when you want it per run, send it with the request:

```bash
quackd run find-and-kick --provider vllm --model Qwen/Qwen3-8B \n  --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
```

That flag is a JSON string, and no single spelling of one survives every shell: the line above
is for bash, PowerShell 5.1 wants `'{\"chat_template_kwargs\": {\"enable_thinking\": false}}'`,
and `cmd.exe` wants the whole thing in double quotes with the inner ones escaped. The way round
all of it is a line in `.env`, which every shell leaves alone:

```
QUACKD_EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
```

Single quotes there, or none. Double quotes around JSON make python-dotenv drop the variable
without setting it, and the run then thinks out loud as though you had never written the line.

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
| `--extra-body` / `QUACKD_EXTRA_BODY` | one JSON object, merged into the top of every request body | nothing extra is sent |

`parallel_tool_calls` is never sent to local servers, because some reject unknown fields, and
nothing else is added unless `--extra-body` asks for it.

`--extra-body` works on every provider that speaks OpenAI's API, which is nine of the eleven
cloud vendors and all five local presets, and on Chat Completions and Responses alike, so it
keeps working when a run moves from one to the other. The flag beats the variable, and an empty
object sends nothing, which is how a `.env` line is silenced for a single run. Six keys are
refused because they are quackd's to send: `model`, `messages`, `input`, `instructions`,
`tools` and `stream`. The odd one there is `instructions`, which is the system prompt on the
Responses API the way `messages` carries it on Chat Completions. Everything else replaces what
quackd would have sent, `tool_choice` included, because overriding it is the point. That cuts
both ways: `n` or `response_format` will reach the server too, and what the model answers with
afterwards is yours to live with. In a flock the object goes to every member that speaks
OpenAI's API, and there is no per robot value in the registry.

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
and it is also what you run when you are changing the page. `web/serve.py` is the server it
needs, stdlib only, so it is Python but not quackd ([web/README.md](../web/README.md) says
why a plain `http.server` will not do):

```bash
python web/serve.py            # then open http://localhost:8000/simulator/
```

If port 8000 is already a vLLM, which the table above assumes it is, then
`python web/serve.py 8001` moves the page rather than the model server.

Either copy is driven the same way. Pick Local in the page and give it your base URL. Ollama has
to be told to accept the page (`OLLAMA_ORIGINS=* ollama serve`), and llama.cpp, vLLM and LM
Studio need the same CORS permission, whichever copy you opened. The keyboard beside the sentence
box is live at the same time as the model, so you can take the duck off a stalled local model
mid-run with `W` and the transcript records the handover.
What the page has and has not been run against is in [web/README.md](../web/README.md).

## Honest notes

- Which local model pilots the duck well is an open question we have barely measured. The
  loop was designed so that a weak planner degrades the task, never the robot's balance.
  Two transcripts are published here and they are not ours: the contributor who built
  memory between runs ran `find-and-kick` against **Qwen 2.5 Coder 14B on LM Studio**
  (Apple M2 Pro, 2026-09-03) and they are in [`assets/transcripts/`](assets/transcripts/).
  They are two runs out of more than two: [docs/design/memory.md](design/memory.md)
  records the same model reading the memory block and never writing to it *across four
  runs* before `remember` was moved into the numbered strategy. These two were kept, so
  read them as a selection rather than as the sample:

  | transcript | seed | outcome | steps | LLM calls | tokens in + out | text fallbacks | what it shows |
  |---|---|---|---|---|---|---|---|
  | [`…seed6-memory-read.jsonl`](assets/transcripts/qwen2.5-coder-14b-lmstudio-find-and-kick-seed6-memory-read.jsonl) | 6 | success | 8 | 9 | 29,403 + 244 | 0 | the system prompt carries an earlier run's episode under *What you remember*; the model never calls `remember`; two kicks fall short before the third connects |
  | [`…seed5-remember.jsonl`](assets/transcripts/qwen2.5-coder-14b-lmstudio-find-and-kick-seed5-remember.jsonl) | 5 | success | 4 | 6 | 17,939 + 265 | 0 | the `.duck` body now says `remember` in strategy step 5; after the kick the model returns `remember`, `quack` and `declare_success` in one response, the loop keeps the first (a fact from the verb results) and marks `multiple_tool_calls`, and the other two arrive one per turn after |

  Two more arrived on 2026-09-14, from the contributor who asked for `--extra-body`, and
  they are a pair rather than a sample: the same build (`739ff84`), the same seed, the same
  server, one variable. **Qwen3-32B-AWQ on vLLM 0.27.2.dev**, on an NVIDIA GB10, which is
  `aarch64` and not a machine this project has ever run on.

  | transcript | seed | outcome | steps | LLM calls | tokens in + out | text fallbacks | what it shows |
  |---|---|---|---|---|---|---|---|
  | [`…seed1-thinking-on.jsonl`](assets/transcripts/qwen3-32b-awq-vllm-find-and-kick-seed1-thinking-on.jsonl) | 1 | success | 4 | 8 | 35,416 + 2,049 | 0 | Qwen3 with its factory default: every one of the eight calls opens with visible deliberation, 165 to 402 output tokens each |
  | [`…seed1-thinking-off.jsonl`](assets/transcripts/qwen3-32b-awq-vllm-find-and-kick-seed1-thinking-off.jsonl) | 1 | success | 3 | 5 | 21,802 + 263 | 0 | the same run with `--extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'`: no deliberation anywhere, 19 to 130 output tokens a call |

  **They did not do the same work, so read the per-call numbers and not the totals.** The
  thinking run took four steps and spent two of them on `remember` and `quack`; the quiet
  one took three and went from the kick to the declaration. Part of 2,049 → 263 is a shorter
  path. What the path cannot explain is 165 to 402 tokens a call becoming 19 to 130.

  The first call stays expensive either way: 130 tokens and 16.0 s with thinking off, against
  19 to 40 tokens and 1.9 to 3.9 s for every call after it. Whatever that is, it is not
  deliberation, because the quiet transcript contains none.

  `reasoning_tokens` reads 0 in both, and that is a property of the server rather than the
  model: without `--reasoning-parser` vLLM leaves the thinking inside `content`, where it is
  billed as output. Output tokens is therefore the honest column here, and a run that reports
  no reasoning tokens is not a run that did no reasoning.

  The simulator clock says 6.7 s for both, because the robot did the same thing at the same
  speed; the transcript timestamps say 195.9 s and 29.6 s of wall clock. On this server the
  same result is available at serve time, with `--reasoning-parser qwen3` beside
  `--default-chat-template-kwargs '{"enable_thinking": false}'`, and that is the better answer
  for a box you own. The flag is what you have on one you do not.

  These two are not a chain, and nothing here should be read as one: they ran against
  different memory directories (`memory-qwen3` and `memory-qwen2`), seed 5 started from an
  empty memory block, and the episode seed 6 remembers was written by a run that is not in
  this repository. Seed 5 shows the write and seed 6 shows the read. Neither shows the
  other's half.

  Every turn was a native tool call, none needed the JSON text fallback. The simulator
  clock (`elapsed_s` in `run_end`, which is what the budget counts on `sim2d`) says 14 s and
  11 s; the transcript timestamps say 33 s and 29 s of wall clock, three to nine seconds per
  LLM call. What they cannot show: anything about another model, another machine, or a
  harder task than the starter duck, and neither one shows a note surviving from the run
  that wrote it into the run that reads it. If you run one, please
  share the transcript in a Discussion or a PR into that folder: it is the cheapest way
  to make this section shorter.
- The cloud providers keep their stricter settings (`tool_choice="required"`,
  `parallel_tool_calls=False`). Only the local presets use the relaxed ones.
- Ollama, vLLM, llama.cpp and LM Studio evolve quickly. If a flag above is stale, open an
  issue with the server version.
