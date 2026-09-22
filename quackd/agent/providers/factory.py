"""`--llm <vendor>[:<model>]` to an `LLMProvider`, importing vendor SDKs only when asked for.

Kept separate from `__init__` so importing `quackd.agent.providers` never touches a vendor
package, and so `quackd doctor` can ask "which providers could run here?" cheaply.

`make_provider` reads no environment variable at all. `resolve_llm` is the single place that
knows the order of precedence (a flag beats a registered robot beats `QUACKD_LLM` beats the
default), and it hands the factory an answer that is already decided, along with the phrase
naming where that answer came from. When the factory also consulted the environment, two callers
passing identical arguments could get different pilots, and the error a bad id raised could not
say which of the two the reader needed to fix.

The model names themselves live one module further out, in `catalogue`, which imports nothing at
all: the CLI reads it to build `--llm`'s completions and its help, and must not pay for pydantic
to do that.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from quackd.agent.providers.base import LLMProvider, ProviderError
from quackd.agent.providers.catalogue import CATALOGUE as CATALOGUE
from quackd.agent.providers.catalogue import CLOUD_NAMES as CLOUD_NAMES
from quackd.agent.providers.catalogue import DEFAULT_LLM as DEFAULT_LLM
from quackd.agent.providers.catalogue import LLM_ENV as LLM_ENV
from quackd.agent.providers.catalogue import LOCAL_NAMES as LOCAL_NAMES
from quackd.agent.providers.catalogue import PROVIDER_NAMES as PROVIDER_NAMES
from quackd.agent.providers.catalogue import default_model_for as default_model_for
from quackd.agent.providers.catalogue import find_model as find_model
from quackd.agent.providers.catalogue import model_ids as model_ids
from quackd.agent.providers.catalogue import models_for as models_for
from quackd.agent.providers.catalogue import vendor_of as vendor_of

# The default for each provider, derived from the catalogue so an id is spelled once. Local
# presets have no default: they discover the served model from /v1/models when none is given.
DEFAULT_MODELS: dict[str, str | None] = {
    **{name: default_model_for(name) for name in CLOUD_NAMES},
    **{name: None for name in LOCAL_NAMES},
}

KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "grok": "XAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "cohere": "COHERE_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "glm": "ZAI_API_KEY",
    "meta": "META_API_KEY",
    **{name: "LOCAL_API_KEY" for name in LOCAL_NAMES},
}

# Which `quackd[...]` extra installs the SDK a provider needs. Most of these are the same wheel
# under a different name, because most vendors speak OpenAI's API: the extra exists so the error
# a missing package raises can name the install the reader actually wants.
EXTRA_FOR = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "gemini",
    "grok": "grok",
    "mistral": "mistral",
    "deepseek": "deepseek",
    "cohere": "cohere",
    "qwen": "qwen",
    "kimi": "kimi",
    "glm": "glm",
    "meta": "meta",
    **{name: "openai" for name in LOCAL_NAMES},
}

# The module `doctor` imports to decide whether a provider could run here. Only three distinct
# SDKs serve eleven vendors and five local presets.
SDK_FOR = {
    "anthropic": "anthropic",
    "gemini": "google.genai",
    **{name: "openai" for name in CLOUD_NAMES if name not in ("anthropic", "gemini")},
    **{name: "openai" for name in LOCAL_NAMES},
}

# Vendors that are OpenAI's API wearing a different hat: a base URL, a key and a model list.
# Each is one small subclass in `quackd/agent/providers/<name>.py`, imported only when asked for.
OPENAI_COMPATIBLE = {
    "grok": "GrokProvider",
    "mistral": "MistralProvider",
    "deepseek": "DeepSeekProvider",
    "cohere": "CohereProvider",
    "qwen": "QwenProvider",
    "kimi": "KimiProvider",
    "glm": "GLMProvider",
    "meta": "MetaProvider",
}


def _unknown_model(provider: str, model: str, source: str) -> str:
    """Why an id was refused, and what to pass instead.

    Every part of this earns its place. A model from another vendor is the commonest mistake and
    the hardest to see, because the id looks perfectly valid. The full list of ids is here rather
    than behind a command because the reader is already stopped.
    """
    ids = model_ids(provider)
    default = ids[0] if ids else ""
    listed = ", ".join(f"{i} (default)" if i == default else i for i in ids)
    elsewhere = vendor_of(model)
    whose = (
        f" ({model!r} is a {elsewhere} model: --llm {elsewhere}:{model})"
        if elsewhere and elsewhere != provider
        else ""
    )
    return (
        f"{provider}: unknown model {model!r} from {source}{whose}. "
        f"Valid ids: {listed}. See `quackd list-models --llm {provider}`."
    )


def resolve_model(provider: str, model: str | None, *, source: str = "--llm") -> str | None:
    """The id this provider will be given, or a `ProviderError` saying why not.

    Cloud vendors take an id from the catalogue and nothing else, so a retired id, a typo and
    another vendor's id all stop here, before a key is read or a packet is sent. Everything
    without a catalogue passes straight through: the local presets serve whatever was pulled, and
    `None` there means "ask the server" rather than "use the default" (ADR-0014).
    """
    if provider not in CATALOGUE:
        return model
    if not model:
        return default_model_for(provider)
    if find_model(provider, model) is not None:
        return model
    raise ProviderError(_unknown_model(provider, model, source))


def _unknown_llm(spec: str, head: str, source: str, *, bare: bool) -> str:
    """Why a vendor name was refused, and both shapes of the flag that would have worked.

    Two mistakes land here and they want different words. `--llm hal:gpt-4o` is a typo in the
    vendor half, so the message names the whole spec back: that is what shows the reader which
    half of it quackd could not read. `--llm hal9000` might instead have been meant as a bare
    model id, which is a form that does work (`--llm claude-opus-5`), so that case also says the
    catalogue was searched and came back empty, rather than leaving the reader to wonder whether
    bare ids are allowed at all.
    """
    where = f" in {spec!r}" if spec != head else ""
    searched = ", and no vendor here lists a model of that name" if bare else ""
    example = default_model_for("anthropic")
    return (
        f"unknown provider {head!r}{where} from {source}{searched}. "
        f"Pass a vendor, or a vendor and a model: --llm anthropic, --llm anthropic:{example}. "
        f"Vendors: {', '.join(PROVIDER_NAMES)}."
    )


def parse_llm(
    spec: str | None, *, source: str = "--llm", check_model: bool = True
) -> tuple[str, str | None]:
    """`--llm` as the (vendor, model) pair the factory takes, or a `ProviderError` saying why not.

    One flag now carries what `--provider` and `--model` used to carry between them, because the
    two were never really independent: a model id means nothing without its vendor, and every
    refusal had to name both anyway. `--llm anthropic` is that vendor's default model,
    `--llm anthropic:claude-sonnet-5` names one, and no `--llm` at all is `fake`.

    The split is at the FIRST colon and no other, because Ollama's own tags contain one:
    `--llm ollama:llama3:8b` is the preset `ollama` serving the model `llama3:8b`, where a split
    on the last colon would have asked it for `llama3`. The vendor half is lowercased, so
    `--llm OpenAI` works; the model half is not, because vendors ship ids like `Qwen/Qwen3-8B`
    and a folded copy of one is a 404.

    A spec with no colon that is not a vendor gets one more chance. Catalogue ids are unique
    across vendors (`vendor_of`, and a test that holds them to it), so `--llm claude-opus-5`
    can infer `anthropic` on its own and spare the reader remembering which house builds what.

    Pass `check_model=False` to learn only which vendor was named, without the catalogue lookup:
    shell completion has to answer while the id after the colon is still half typed.
    """
    if spec is None or not spec.strip():
        return DEFAULT_LLM, None
    text = spec.strip()
    head, colon, tail = text.partition(":")
    head = head.strip()
    vendor = head.lower()
    model = tail.strip() or None
    if vendor in PROVIDER_NAMES:
        # A colon with nothing after it is not a mistake. Shell completion offers `openai:` as a
        # prefix, and a reader who presses enter on it means the vendor's default, not an error.
        if model is not None and check_model:
            resolve_model(vendor, model, source=source)
        return vendor, model
    if not colon:
        inferred = vendor_of(head)
        if inferred is not None:
            return inferred, head
    raise ProviderError(_unknown_llm(text, head, source, bare=not colon))


def resolve_llm(
    flag: str | None, stored: str | None = None, *, robot: str | None = None
) -> tuple[str, str | None, str]:
    """Which pilot to fly, from the three places one can be named, and which place named it.

    The order is the one every other quackd setting uses: what the reader has just typed beats
    what a registered robot remembers, which beats the environment, which beats `fake`. The
    source phrase travels back with the answer so a bad value is refused in the reader's own
    terms. "unknown provider 'hal' from robot duck-a (robots.json)" sends them to a line in a
    file they may have written months ago, while the same text from `--llm` sends them to the
    command still on their screen, and the two are a very different hunt.

    Blank counts as absent at every level: `QUACKD_LLM=` in a `.env` is how a shell says "unset",
    and reading that as a vendor named "" would stop the run instead of falling through to the
    next place.
    """
    where = f"robot {robot} (robots.json)" if robot else "robots.json"
    candidates: tuple[tuple[str | None, str], ...] = (
        (flag, "--llm"),
        (stored, where),
        (os.environ.get(LLM_ENV), LLM_ENV),
    )
    for value, source in candidates:
        if value is None or not value.strip():
            continue
        vendor, model = parse_llm(value, source=source)
        return vendor, model, source
    return DEFAULT_LLM, None, "default"


def _extra_body(text: str | None) -> dict[str, Any] | None:
    """`--extra-body` as the dict a provider takes, parsed here so a bad value names the flag
    and stops before a key is read or a packet is sent. None hands the provider nothing, and it
    reads `QUACKD_EXTRA_BODY` itself: that is how the flag outranks the variable."""
    if text is None:
        return None
    from quackd.agent.providers.openai import parse_extra_body

    return parse_extra_body(text, source="--extra-body")


def make_provider(
    name: str,
    *,
    model: str | None = None,
    source: str = "--llm",
    duck_name: str | None = None,
    goal: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    vision: bool | None = None,
    extra_body: str | None = None,
) -> LLMProvider:
    name = name.lower()
    # Before the branches, so a typo is refused the same way whichever provider was named,
    # `fake` included. Anthropic and Gemini ignore the value as they ignore `--base-url`,
    # but ignoring a field is not the same as swallowing a mistake, and `--llm fake`
    # is then the cheapest way to find out whether a shell mangled the quoting.
    body = _extra_body(extra_body)
    if name == "fake":
        from quackd.agent.providers.fake import FakeProvider

        # The scripted pilot has no model to pick, so `--llm fake:anything` is not refused here,
        # the model half is ignored. `--vision` it does take: it looks at nothing either way, and
        # it is the only pilot that can carry a picture through the whole loop with no key and no
        # vendor.
        return FakeProvider.for_duck(duck_name or "", goal=goal, vision=vision)
    # No environment is read here: `resolve_llm` has already settled what was asked for, and
    # `source` says where it was asked, so a wrong id names the flag the reader has just typed
    # or the line in `robots.json` they had forgotten, rather than guessing between them.
    model = resolve_model(name, model or None, source=source)
    if name == "anthropic":
        from quackd.agent.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model=model, vision=vision)
    if name == "openai":
        from quackd.agent.providers.openai import OpenAIProvider

        return OpenAIProvider(
            model=model,
            api_key=api_key,
            base_url=base_url,
            vision=vision,
            extra_body=body,
        )
    if name == "gemini":
        from quackd.agent.providers.gemini import GeminiProvider

        return GeminiProvider(model=model, api_key=api_key, vision=vision)
    if name in OPENAI_COMPATIBLE:
        module = importlib.import_module(f"quackd.agent.providers.{name}")
        vendor = getattr(module, OPENAI_COMPATIBLE[name])
        provider: LLMProvider = vendor(
            model=model,
            api_key=api_key,
            base_url=base_url,
            vision=vision,
            extra_body=body,
        )
        return provider
    if name in LOCAL_NAMES:
        from quackd.agent.providers.local import LocalProvider

        return LocalProvider(
            model,
            preset=name,
            base_url=base_url,
            api_key=api_key,
            vision=vision,
            extra_body=body,
        )
    raise ProviderError(f"unknown provider {name!r}; choose one of {', '.join(PROVIDER_NAMES)}")
