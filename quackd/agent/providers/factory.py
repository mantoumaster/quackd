"""`--provider <name>` and `--model <id>` to an `LLMProvider`, importing vendor SDKs only when
asked for.

Kept separate from `__init__` so importing `quackd.agent.providers` never touches a vendor
package, and so `quackd doctor` can ask "which providers could run here?" cheaply.

The model names themselves live one module further out, in `catalogue`, which imports nothing at
all: the CLI reads it to build `--model`'s completions and its help, and must not pay for pydantic
to do that.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from quackd.agent.providers.base import LLMProvider, ProviderError
from quackd.agent.providers.catalogue import CATALOGUE as CATALOGUE
from quackd.agent.providers.catalogue import CLOUD_NAMES as CLOUD_NAMES
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


def default_model(provider: str) -> str | None:
    """What this provider would use with no `--model`, `QUACKD_MODEL` included and unchecked.

    `resolve_model` is what decides whether that answer is allowed. This is the raw wish, which is
    what `quackd doctor` wants to show even when the wish is wrong.
    """
    return os.environ.get("QUACKD_MODEL") or DEFAULT_MODELS.get(provider)


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
        f" ({model!r} is a {elsewhere} model: pass --provider {elsewhere})"
        if elsewhere and elsewhere != provider
        else ""
    )
    return (
        f"{provider}: unknown model {model!r} from {source}{whose}. "
        f"Valid ids: {listed}. See `quackd list-models --provider {provider}`."
    )


def resolve_model(provider: str, model: str | None, *, source: str = "--model") -> str | None:
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
    # but ignoring a field is not the same as swallowing a mistake, and `--provider fake`
    # is then the cheapest way to find out whether a shell mangled the quoting.
    body = _extra_body(extra_body)
    if name == "fake":
        from quackd.agent.providers.fake import FakeProvider

        # The scripted pilot has no model to pick, so `--model` is not refused here, it is ignored.
        # `--vision` it does take: it looks at nothing either way, and it is the only pilot
        # that can carry a picture through the whole loop with no key and no vendor.
        return FakeProvider.for_duck(duck_name or "", goal=goal, vision=vision)
    # Named before it is resolved, so the error can say where a wrong id came from: a flag the
    # reader has just typed and a line in a `.env` they have forgotten want different answers.
    source = "--model" if model else "QUACKD_MODEL"
    model = resolve_model(name, model or os.environ.get("QUACKD_MODEL") or None, source=source)
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
