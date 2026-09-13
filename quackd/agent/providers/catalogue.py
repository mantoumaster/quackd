"""The models quackd will let you pick, one curated list per cloud vendor.

`--model` used to take any string and hand it straight to the vendor, so a typo, a model retired
last spring and a model belonging to a different vendor all failed the same way: at the first
call, in the vendor's words, after the run had already started. Three of the four defaults quackd
shipped were wrong by the time anyone checked them, which is what a field nobody validates does.

So the list lives here instead, and a user picks from it: on the CLI, in shell completion, in the
browser demo, and in `quackd list-models`. When a vendor ships a model it is added here and
nowhere else. That is the cost of the promise. The list is only as current as its last edit, and
`quackd list-models` is how anyone sees what this build knows.

What earns a place, checked 2026-09-12 against each vendor's own documentation:

- callable that day on the vendor's public API by anyone holding a key,
- a text model that can call function tools, because every verb quackd has is a function tool and
  a model without them is not a degraded pilot, it is no pilot,
- not deprecated, where an announced shutdown date is enough to keep it out,
- not behind an approval programme, because that only buys the reader a 403.

Image, audio, video, embedding and OCR models are not here. Open-weight models are, when the
vendor serves them on its own API rather than only publishing the weights, and so are models a
vendor hosts but did not train, because what matters is that the vendor answers for them.

The local presets are deliberately absent. They serve whatever you pulled, so `--model` stays free
text there and quackd asks the server what it has (ADR-0014).

Nothing here may import a vendor SDK, pydantic, or anything else heavy: `quackd --help` imports
this module, and so does every press of TAB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

Status = Literal["current", "legacy", "preview", "specialised", "open"]

#: Every status there is, in the order a list of models should be shown in. `list-models` prints
#: them in this order and the browser demo groups its dropdown by it.
STATUSES: tuple[Status, ...] = get_args(Status)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model a user may pick, as the vendor spells it."""

    id: str
    """The API id, character for character. This is what goes on the wire."""

    label: str
    """What a human reads in a dropdown or a table."""

    status: Status = "current"
    """`current` is the vendor's headline lineup, `legacy` still served with no end announced,
    `preview` the vendor's own label, `specialised` tuned for one thing (code, a deep reasoning
    tier, vision, multi-agent, a pricing tier), `open` an open-weight model the vendor serves."""

    vision: bool = True
    """Does the vendor document image input? Where it does not, quackd keeps the camera frame to
    itself and sends the detections as text, the way it already does for local models. `--vision`
    and `--no-vision` override this in both directions."""

    api: Literal["chat", "responses"] | None = None
    """OpenAI only, and only when the model needs saying. Some models will not take function tools
    on Chat Completions at all and name `/v1/responses` in the 400. `openai.py` can read that 400
    and move, but it pays a failed call to learn it, and a model that is Responses only answers
    with a different error that the 400 reader does not match. Naming the API here costs nothing
    and skips both problems. `None` means start on Chat Completions, as everything else does."""


# ── the catalogue ───────────────────────────────────────────────────────────────────────
#
# Insertion order is display order, and THE FIRST ENTRY OF EACH VENDOR IS ITS DEFAULT: the model
# `--model` means when it is left off. Writing the default as a separate field would spell the
# same id twice and let the two drift.

CATALOGUE: dict[str, tuple[ModelSpec, ...]] = {
    # platform.claude.com models overview and model-deprecations. Claude Mythos is invitation only
    # (Project Glasswing). Opus 4.1 and everything 4.0 or older is retired. Three of these ids are
    # aliases onto a dated snapshot (haiku-4-5, opus-4-5, sonnet-4-5); the rest are the snapshot
    # itself, and appending a date to one of those is an error rather than a pin.
    "anthropic": (
        ModelSpec("claude-opus-5", "Claude Opus 5"),
        ModelSpec("claude-fable-5-1", "Claude Fable 5.1"),
        ModelSpec("claude-sonnet-5", "Claude Sonnet 5"),
        ModelSpec("claude-haiku-4-5", "Claude Haiku 4.5"),
        ModelSpec("claude-fable-5", "Claude Fable 5", "legacy"),
        ModelSpec("claude-opus-4-8", "Claude Opus 4.8", "legacy"),
        ModelSpec("claude-opus-4-7", "Claude Opus 4.7", "legacy"),
        ModelSpec("claude-opus-4-6", "Claude Opus 4.6", "legacy"),
        ModelSpec("claude-opus-4-5", "Claude Opus 4.5", "legacy"),
        ModelSpec("claude-sonnet-4-6", "Claude Sonnet 4.6", "legacy"),
        ModelSpec("claude-sonnet-4-5", "Claude Sonnet 4.5", "legacy"),
    ),
    # developers.openai.com models and deprecations. The whole gpt-5 and o-series generation has
    # shutdown dates in 2026, so none of it is here. The Daybreak, Cyber and Rosalind models need
    # their own approval. `gpt-oss-*` are weights, not an endpoint.
    "openai": (
        ModelSpec("gpt-5.6-sol", "GPT-5.6 Sol"),
        ModelSpec("gpt-6-astra", "GPT-6 Astra", api="responses"),
        ModelSpec("gpt-5.6-terra", "GPT-5.6 Terra"),
        ModelSpec("gpt-5.6-luna", "GPT-5.6 Luna"),
        ModelSpec("gpt-5.5", "GPT-5.5", "legacy"),
        ModelSpec("gpt-5.4", "GPT-5.4", "legacy"),
        ModelSpec("gpt-5.4-mini", "GPT-5.4 mini", "legacy"),
        ModelSpec("gpt-5.4-nano", "GPT-5.4 nano", "legacy"),
        ModelSpec("gpt-5.2", "GPT-5.2", "legacy"),
        ModelSpec("gpt-5.1", "GPT-5.1", "legacy"),
        ModelSpec("gpt-4.1", "GPT-4.1", "legacy"),
        ModelSpec("gpt-4.1-mini", "GPT-4.1 mini", "legacy"),
        ModelSpec("gpt-4o", "GPT-4o", "legacy"),
        ModelSpec("gpt-4o-mini", "GPT-4o mini", "legacy"),
        ModelSpec("gpt-5.5-pro", "GPT-5.5 Pro", "specialised", api="responses"),
        ModelSpec("gpt-5.4-pro", "GPT-5.4 Pro", "specialised", api="responses"),
        ModelSpec("gpt-5.2-pro", "GPT-5.2 Pro", "specialised", api="responses"),
        ModelSpec("gpt-5.3-codex", "GPT-5.3 Codex", "specialised", api="responses"),
        ModelSpec("chat-latest", "ChatGPT Instant (moves with ChatGPT)", "specialised"),
    ),
    # ai.google.dev models and deprecations. Gemini 3.1 Flash-Lite has a shutdown date already,
    # the 2.0 family and the Gemini 3 Pro preview are gone, and Robotics ER 1.6 went on
    # 2026-08-31. ER 2 is here because quackd drives robots and that is its subject. Google's own
    # page calls 3.5 Flash "our legacy Flash model", so that is where it sits.
    "gemini": (
        ModelSpec("gemini-3.8-flash", "Gemini 3.8 Flash"),
        ModelSpec("gemini-3.7-flash", "Gemini 3.7 Flash"),
        ModelSpec("gemini-3.6-flash", "Gemini 3.6 Flash"),
        ModelSpec("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite"),
        ModelSpec("gemini-3.1-pro-preview", "Gemini 3.1 Pro (preview)", "preview"),
        ModelSpec("gemini-3-flash-preview", "Gemini 3 Flash (preview)", "preview"),
        ModelSpec("gemini-robotics-er-2-preview", "Gemini Robotics ER 2 (preview)", "preview"),
        ModelSpec("gemini-3.5-flash", "Gemini 3.5 Flash", "legacy"),
        ModelSpec("gemini-2.5-pro", "Gemini 2.5 Pro", "legacy"),
        ModelSpec("gemini-2.5-flash", "Gemini 2.5 Flash", "legacy"),
        ModelSpec("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite", "legacy"),
    ),
    # docs.x.ai. grok-4 and the whole grok-3 and grok-4-fast line were retired on 2026-05-15 and
    # now answer as grok-4.3 without saying so, which is the kind of silence a catalogue exists
    # to stop. All seven take an image and all seven do function calling.
    "grok": (
        ModelSpec("grok-4.6", "Grok 4.6"),
        ModelSpec("grok-4.5", "Grok 4.5"),
        ModelSpec("grok-4.3", "Grok 4.3"),
        ModelSpec("grok-4.20-0309-reasoning", "Grok 4.20 (reasoning)", "legacy"),
        ModelSpec("grok-4.20-0309-non-reasoning", "Grok 4.20 (non-reasoning)", "legacy"),
        ModelSpec("grok-4.20-multi-agent-0309", "Grok 4.20 multi-agent", "specialised"),
        ModelSpec("grok-build-0.1", "Grok Build 0.1", "specialised"),
    ),
    # docs.mistral.ai. Magistral retired on 2026-07-31, and Devstral and Pixtral are deprecated.
    # Medium 3.5 is the one id here that does not follow the dated pattern: Mistral's lifecycle
    # page has moved to name-major-minor and the model card lists `mistral-medium-3-5`. The last
    # two are models Mistral serves but did not train, at its own address under its own ids.
    "mistral": (
        ModelSpec("mistral-medium-3-5", "Mistral Medium 3.5"),
        ModelSpec("mistral-large-2512", "Mistral Large 3"),
        ModelSpec("mistral-small-2603", "Mistral Small 4"),
        ModelSpec("ministral-14b-2512", "Ministral 3 14B", "open"),
        ModelSpec("ministral-8b-2512", "Ministral 3 8B", "open"),
        ModelSpec("ministral-3b-2512", "Ministral 3 3B", "open"),
        ModelSpec("codestral-2508", "Codestral 25.08", "specialised", vision=False),
        ModelSpec("zai-glm-5-2", "Z.ai GLM 5.2 (hosted by Mistral)", "preview", vision=False),
        ModelSpec(
            "labs-leanstral-1-5",
            "Leanstral 1.5 (Lean 4 proofs, Mistral labs)",
            "preview",
            vision=False,
        ),
    ),
    # api-docs.deepseek.com. `deepseek-chat` and `deepseek-reasoner` are retired aliases.
    "deepseek": (
        ModelSpec("deepseek-flash", "DeepSeek V4.1 Flash"),
        ModelSpec("deepseek-v4-pro", "DeepSeek V4 Pro", vision=False),
    ),
    # docs.cohere.com, reached through the OpenAI compatibility endpoint. Command A Translate and
    # Command A Vision are out for the same reason, which their own pages state: no tool use. The
    # North models are out because nothing in their docs says they have any either.
    "cohere": (
        ModelSpec("command-a-plus-05-2026", "Command A+"),
        ModelSpec("command-a-03-2025", "Command A", vision=False),
        ModelSpec(
            "command-a-reasoning-08-2025", "Command A Reasoning", "specialised", vision=False
        ),
        ModelSpec("command-r-plus-08-2024", "Command R+", "legacy", vision=False),
        ModelSpec("command-r-08-2024", "Command R", "legacy", vision=False),
        ModelSpec("command-r7b-12-2024", "Command R7B", "legacy", vision=False),
    ),
    # Alibaba Cloud Model Studio. Only the rolling ids are here: the dated snapshots behind them
    # (`qwen3.8-max-0902` and the rest) are a pinning mechanism, not a menu, and Alibaba gives them
    # a month of notice against three for a rolling id. Qwen3.7 Max sits under the vendor's own
    # "Legacy models" heading even though 3.7 Plus and Flash do not.
    "qwen": (
        ModelSpec("qwen3.8-max", "Qwen3.8 Max"),
        ModelSpec("qwen3.8-flash", "Qwen3.8 Flash"),
        ModelSpec("qwen3.7-plus", "Qwen3.7 Plus"),
        ModelSpec("qwen3.7-flash", "Qwen3.7 Flash"),
        ModelSpec("qwen3.7-max", "Qwen3.7 Max", "legacy", vision=False),
        ModelSpec("qwen3.6-plus", "Qwen3.6 Plus", "legacy"),
        ModelSpec("qwen3.6-flash", "Qwen3.6 Flash", "legacy"),
        ModelSpec("qwen3.5-plus", "Qwen3.5 Plus", "legacy"),
        ModelSpec("qwen3.5-flash", "Qwen3.5 Flash", "legacy"),
        ModelSpec("qwen3-max", "Qwen3 Max", "legacy", vision=False),
        ModelSpec("qwen-plus", "Qwen Plus", "legacy", vision=False),
        ModelSpec("qwen-flash", "Qwen Flash", "legacy", vision=False),
        ModelSpec("qwen3-coder-next", "Qwen3 Coder Next", "specialised", vision=False),
        ModelSpec("qwen3-coder-plus", "Qwen3 Coder Plus", "specialised", vision=False),
        ModelSpec("qwen3-coder-flash", "Qwen3 Coder Flash", "specialised", vision=False),
        ModelSpec("qwen3.8-27b", "Qwen3.8 27B", "open"),
        ModelSpec("qwen3.8-2.4t-a95b", "Qwen3.8 2.4T-A95B", "open", vision=False),
        ModelSpec("qwen3.6-35b-a3b", "Qwen3.6 35B-A3B", "open"),
        ModelSpec("qwen3.6-27b", "Qwen3.6 27B", "open"),
        ModelSpec("qwen3.5-397b-a17b", "Qwen3.5 397B-A17B", "open"),
        ModelSpec("qwen3.5-122b-a10b", "Qwen3.5 122B-A10B", "open"),
        ModelSpec("qwen3.5-27b", "Qwen3.5 27B", "open"),
        ModelSpec("qwen3.5-35b-a3b", "Qwen3.5 35B-A3B", "open"),
    ),
    # platform.kimi.ai. Everything before K2.6 was discontinued on 2026-05-25 and answers 404.
    # Only K3 accepts a forced tool call: the other three have thinking permanently on, and that
    # narrows tool_choice to auto or none, which is why `kimi.py` asks rather than insists.
    "kimi": (
        ModelSpec("kimi-k3", "Kimi K3"),
        ModelSpec("kimi-k2.6", "Kimi K2.6"),
        ModelSpec("kimi-k2.7-code", "Kimi K2.7 Code", "specialised"),
        ModelSpec("kimi-k2.7-code-highspeed", "Kimi K2.7 Code (high speed)", "specialised"),
    ),
    # docs.z.ai. The `v` models are the vision line and the rest are text only. GLM-4.5V is not
    # here: it is the one of those whose page does not give it function calling.
    "glm": (
        ModelSpec("glm-5.3", "GLM-5.3", vision=False),
        ModelSpec("glm-5.3-flash", "GLM-5.3 Flash"),
        ModelSpec("glm-5.2", "GLM-5.2", "legacy", vision=False),
        ModelSpec("glm-5.1", "GLM-5.1", "legacy", vision=False),
        ModelSpec("glm-5", "GLM-5", "legacy", vision=False),
        ModelSpec("glm-4.7", "GLM-4.7", "legacy", vision=False),
        ModelSpec("glm-4.7-flash", "GLM-4.7 Flash", "legacy", vision=False),
        ModelSpec("glm-4.7-flashx", "GLM-4.7 FlashX", "legacy", vision=False),
        ModelSpec("glm-4.6", "GLM-4.6", "legacy", vision=False),
        ModelSpec("glm-4.5", "GLM-4.5", "legacy", vision=False),
        ModelSpec("glm-4.5-x", "GLM-4.5 X", "legacy", vision=False),
        ModelSpec("glm-4.5-air", "GLM-4.5 Air", "legacy", vision=False),
        ModelSpec("glm-4.5-airx", "GLM-4.5 AirX", "legacy", vision=False),
        ModelSpec("glm-4.5-flash", "GLM-4.5 Flash", "legacy", vision=False),
        ModelSpec("glm-4-32b-0414-128k", "GLM-4 32B (128k)", "legacy", vision=False),
        ModelSpec("glm-4.6v", "GLM-4.6V", "specialised"),
        ModelSpec("glm-4.6v-flash", "GLM-4.6V Flash", "specialised"),
        ModelSpec("glm-4.6v-flashx", "GLM-4.6V FlashX", "specialised"),
    ),
    # Meta retired the hosted Llama API in July 2026. What replaced it is the Meta Model API,
    # which is OpenAI shaped and serves Muse Spark. The contributor tier is cheaper because Meta
    # trains on what you send it, which is why the label says so rather than the release notes.
    "meta": (
        ModelSpec("muse-spark-1.3", "Muse Spark 1.3"),
        ModelSpec("muse-spark-1.2", "Muse Spark 1.2", "legacy"),
        ModelSpec("muse-spark-1.1", "Muse Spark 1.1", "legacy"),
        ModelSpec(
            "muse-spark-1.3-contributor",
            "Muse Spark 1.3 (contributor tier, Meta trains on your prompts)",
            "specialised",
        ),
        ModelSpec(
            "muse-spark-1.2-contributor",
            "Muse Spark 1.2 (contributor tier, Meta trains on your prompts)",
            "specialised",
        ),
    ),
}

#: Vendors with a catalogue. Derived, so adding a vendor above is the whole edit.
CLOUD_NAMES: tuple[str, ...] = tuple(CATALOGUE)

#: Servers that speak OpenAI's API, usually on your own machine. They serve whatever you pulled,
#: so they have no catalogue and `--model` stays free text there (ADR-0014). `local.PRESETS` holds
#: the addresses, and a test pins the two to the same set.
LOCAL_NAMES: tuple[str, ...] = ("local", "ollama", "vllm", "llamacpp", "lmstudio")

PROVIDER_NAMES: tuple[str, ...] = ("fake", *CLOUD_NAMES, *LOCAL_NAMES)


def models_for(provider: str) -> tuple[ModelSpec, ...]:
    """Every model a provider offers, in display order.

    Empty for `fake`, for the local presets and for anything unknown, all of which pick their
    model some other way. A caller that gets `()` must not conclude the provider is broken.
    """
    return CATALOGUE.get(provider, ())


def model_ids(provider: str) -> tuple[str, ...]:
    return tuple(m.id for m in models_for(provider))


def default_model_for(provider: str) -> str | None:
    """The model `--model` means when it is left off, or None where quackd does not choose one."""
    models = models_for(provider)
    return models[0].id if models else None


def find_model(provider: str, model_id: str) -> ModelSpec | None:
    for m in models_for(provider):
        if m.id == model_id:
            return m
    return None


def vendor_of(model_id: str) -> str | None:
    """Which vendor lists this id, if any.

    Ids are unique across the catalogue (a test holds them to it), so this is what turns "unknown
    model" into "that is a grok model, pass --provider grok", which is the mistake worth catching.
    """
    for provider, models in CATALOGUE.items():
        if any(m.id == model_id for m in models):
            return provider
    return None
