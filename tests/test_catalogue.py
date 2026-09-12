"""The model catalogue, and the entries around it that have to agree.

The catalogue is data, and data rots quietly: an id nobody validates is wrong for months, which
is exactly how quackd came to ship `gpt-5`, `grok-4` and `gemini-2.5-pro` as defaults after all
three had been deprecated or retired. Nothing here can tell whether an id is still real, because
that needs the network and a key and neither is available in CI. What it can do is hold the shape:
one default per vendor and it is in its own list, ids unique so a wrong one can be traced to its
owner, a status from the fixed set, and a row in every table a new vendor has to appear in.

`CONTRIBUTING.md` used to say "nothing counts them" about those tables. This file is what
replaced that sentence.
"""

from __future__ import annotations

import importlib
import tomllib
from types import SimpleNamespace as NS
from typing import Any

import pytest

from quackd.agent.providers import catalogue as cat
from quackd.agent.providers.base import ProviderError, ProviderMissingKey
from quackd.agent.providers.factory import (
    CLOUD_NAMES,
    EXTRA_FOR,
    KEY_ENV,
    LOCAL_NAMES,
    OPENAI_COMPATIBLE,
    PROVIDER_NAMES,
    SDK_FOR,
    default_model_for,
    make_provider,
    model_ids,
    models_for,
    resolve_model,
    vendor_of,
)
from quackd.agent.providers.local import PRESETS
from tests.conftest import REPO

#: Ids quackd shipped, or nearly shipped, that the vendor has since retired or that never existed.
#: Kept as a list rather than a rule because there is no rule: the only thing that stops one of
#: these coming back in a careless edit is a test that names them.
GONE = {
    "gpt-5",  # shutdown announced for 2026-12-11
    "grok-4",  # retired 2026-05-15, and silently answered as grok-4.3 after
    "gemini-2.0-flash",  # shut down
    "deepseek-chat",  # retired alias
    "kimi-k2.5",  # discontinued 2026-05-25
    "mistral-medium-2604",  # never an id: Mistral moved to name-major-minor
    "command-a-vision-07-2025",  # live, but its page says tool use is not supported
    "glm-4.5v",  # the one GLM vision model with no function calling
    "claude-mythos-5-1",  # invitation only
    "gpt-5.6-cyber",  # its own approval programme
}


class FakeOpenAI:
    """Enough of the SDK client to construct a provider without a key or a network."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

        async def create(**kwargs: Any) -> Any:
            self.kwargs = kwargs
            return NS(choices=[], usage=None)

        self.chat = NS(completions=NS(create=create))
        self.responses = NS(create=create)


# ── the catalogue holds its shape ───────────────────────────────────────────────────────


def test_every_cloud_provider_has_models_and_the_first_one_is_its_default() -> None:
    assert set(cat.CATALOGUE) == set(CLOUD_NAMES)
    for name in CLOUD_NAMES:
        ids = model_ids(name)
        assert ids, f"{name} has an empty catalogue, so --model could never be satisfied"
        assert default_model_for(name) == ids[0]
        assert default_model_for(name) in ids


def test_nothing_without_a_catalogue_claims_a_default() -> None:
    """`fake` is scripted and the local presets ask their server, so neither may name a model."""
    for name in (*LOCAL_NAMES, "fake"):
        assert default_model_for(name) is None
        assert models_for(name) == ()


def test_model_ids_are_unique_across_every_vendor() -> None:
    """What makes "that is a grok model, pass --provider grok" possible, and truthful."""
    seen: dict[str, str] = {}
    for name in CLOUD_NAMES:
        for model_id in model_ids(name):
            assert model_id not in seen, (
                f"{model_id} is listed by both {seen.get(model_id)} and {name}"
            )
            seen[model_id] = name
    for model_id, name in seen.items():
        assert vendor_of(model_id) == name


def test_every_status_is_one_of_the_five() -> None:
    for name in CLOUD_NAMES:
        for m in models_for(name):
            assert m.status in cat.STATUSES, f"{m.id} has status {m.status!r}"


def test_only_openai_names_an_api_and_only_ever_responses() -> None:
    """The hint exists because OpenAI has two APIs. Nobody else here has a second one to pick."""
    for name in CLOUD_NAMES:
        for m in models_for(name):
            if name == "openai":
                assert m.api in (None, "chat", "responses")
            else:
                assert m.api is None, f"{m.id} names an API, and {name} has only one"
    assert any(m.api == "responses" for m in models_for("openai")), (
        "no OpenAI model is marked responses, so the hint is dead code and the tests for it lie"
    )


def test_no_retired_id_has_crept_back() -> None:
    listed = {m.id for name in CLOUD_NAMES for m in models_for(name)}
    assert not (listed & GONE), f"the catalogue lists ids known to be gone: {sorted(listed & GONE)}"


def test_labels_say_something_the_id_does_not() -> None:
    """A dropdown of raw ids is a dropdown nobody can read, so every model carries a label."""
    for name in CLOUD_NAMES:
        for m in models_for(name):
            assert m.label and m.label != m.id


# ── resolving what the user asked for ───────────────────────────────────────────────────


def test_an_unknown_id_is_refused_with_the_list_and_where_to_find_it() -> None:
    with pytest.raises(ProviderError) as e:
        resolve_model("openai", "gpt-nope")
    message = str(e.value)
    assert "openai" in message and "gpt-nope" in message
    assert default_model_for("openai") in message and "(default)" in message
    assert "list-models" in message
    for model_id in model_ids("openai"):
        assert model_id in message, f"{model_id} is valid but the refusal does not offer it"


def _refusal(provider: str, model: str) -> str:
    with pytest.raises(ProviderError) as e:
        resolve_model(provider, model)
    return str(e.value)


def test_another_vendors_id_says_whose_it_is() -> None:
    """The mistake that looks like no mistake: a real id, at the wrong vendor. Telling the reader
    only that it is unknown would send them hunting for a typo that is not there."""
    message = _refusal("openai", "grok-4.6")
    assert "is a grok model" in message and "--provider grok" in message
    # and an id belonging to nobody says nothing about vendors, because there is nothing to say
    assert "is a" not in _refusal("openai", "gpt-nope").split("Valid ids")[0].replace(
        "is a model", ""
    )


def test_the_refusal_names_where_the_id_came_from() -> None:
    assert "--model" in str(_refusal("openai", "nope"))
    with pytest.raises(ProviderError, match="QUACKD_MODEL"):
        resolve_model("openai", "nope", source="QUACKD_MODEL")


def test_no_model_means_the_default_and_local_presets_are_left_alone() -> None:
    assert resolve_model("openai", None) == default_model_for("openai")
    assert resolve_model("openai", "") == default_model_for("openai")
    # free text, both directions: a local server serves what it serves
    assert resolve_model("ollama", "qwen3:8b") == "qwen3:8b"
    # None here means ask the server, which is not the same as "use the default"
    assert resolve_model("ollama", None) is None
    assert resolve_model("local", "anything at all") == "anything at all"


def test_an_unknown_provider_is_not_this_functions_business() -> None:
    """`make_provider` raises "unknown provider" for that, and it must keep winning."""
    assert resolve_model("hal", "whatever") == "whatever"


# ── the factory refuses before it spends anything ───────────────────────────────────────


def test_a_bad_model_is_refused_before_the_key_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Order matters. With the model checked second, a reader with no key would be told about
    the key, fix that, and only then be told the model was wrong all along."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="unknown model"):
        make_provider("openai", model="gpt-nope")


def test_a_pinned_model_from_the_wrong_vendor_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUACKD_MODEL", "claude-opus-5")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError) as e:
        make_provider("openai")
    assert "QUACKD_MODEL" in str(e.value) and "anthropic" in str(e.value)


def test_the_scripted_pilot_ignores_a_model_rather_than_refusing_it() -> None:
    """`--provider fake` has no model to pick, and failing a demo over an unused flag is rude."""
    assert make_provider("fake", model="gpt-nope", duck_name="hello-world").name == "fake"


def test_an_unknown_provider_still_says_unknown_provider() -> None:
    with pytest.raises(ProviderError, match="unknown provider"):
        make_provider("hal9000")


def test_the_local_branch_is_handed_exactly_what_it_was_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Free text reaches a local preset untouched, including an id no cloud vendor would take."""
    recorded: dict[str, Any] = {}

    def _record(self: Any, model: str | None = None, **kwargs: Any) -> None:
        recorded["model"] = model
        recorded.update(kwargs)
        self.name = "vllm"
        self.model = model

    monkeypatch.setattr("quackd.agent.providers.local.LocalProvider.__init__", _record)
    make_provider("vllm", model="Qwen/Qwen3-8B", base_url="http://gpu:8000/v1", vision=True)
    assert recorded == {
        "model": "Qwen/Qwen3-8B",
        "preset": "vllm",
        "base_url": "http://gpu:8000/v1",
        "api_key": None,
        "vision": True,
    }


# ── one file and the entries that have to agree ─────────────────────────────────────────


def test_every_provider_appears_in_every_table() -> None:
    for name in PROVIDER_NAMES:
        if name == "fake":
            continue
        assert name in KEY_ENV, f"{name} has no key variable"
        assert name in EXTRA_FOR, f"{name} has no extra"
        assert name in SDK_FOR, f"{name} has no SDK"
    assert set(CLOUD_NAMES) <= set(cat.CATALOGUE)
    assert set(LOCAL_NAMES) == set(PRESETS)
    assert {"anthropic", "openai", "gemini"} | set(OPENAI_COMPATIBLE) == set(CLOUD_NAMES), (
        "a cloud provider is either hand written or OpenAI shaped, and this one is neither"
    )


def test_only_three_sdks_serve_all_of_them() -> None:
    assert set(SDK_FOR.values()) == {"anthropic", "openai", "google.genai"}


def test_every_extra_exists_in_the_packaging_and_installs_the_right_sdk() -> None:
    """An extra named in an error message but missing from `pyproject.toml` sends the reader to
    an install that fails."""
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    distribution = {"anthropic": "anthropic", "openai": "openai", "google.genai": "google-genai"}
    for name in PROVIDER_NAMES:
        if name == "fake":
            continue
        extra = EXTRA_FOR[name]
        assert extra in extras, f"quackd[{extra}] is promised for {name} and is not declared"
        wanted = distribution[SDK_FOR[name]]
        assert any(req.startswith(wanted) for req in extras[extra]), (
            f"quackd[{extra}] does not install {wanted}, which {name} imports"
        )


def test_a_model_that_takes_no_image_is_not_sent_one() -> None:
    """The camera frame is the one thing a catalogue entry changes about a running provider.

    A model whose vendor documents no image input gets the detections as text instead, the way a
    local model already does, rather than a 400 on the first turn. The flag is a default, not a
    verdict: `--vision` and `--no-vision` still decide when a reader has an opinion.
    """
    from quackd.agent.providers.mistral import MistralProvider

    assert MistralProvider(model="codestral-2508", client=FakeOpenAI()).supports_vision is False
    assert MistralProvider(model="mistral-large-2512", client=FakeOpenAI()).supports_vision is True
    forced = MistralProvider(model="codestral-2508", client=FakeOpenAI(), vision=True)
    assert forced.supports_vision is True, "--vision must still win over the catalogue"
    refused = MistralProvider(model="mistral-large-2512", client=FakeOpenAI(), vision=False)
    assert refused.supports_vision is False, "--no-vision must still win over the catalogue"


def test_every_vendor_agrees_with_itself_about_images() -> None:
    """Whatever a vendor's entry claims, the provider it builds must report the same thing, or
    `quackd list-models` and the run would disagree about what gets sent."""
    for name in sorted(OPENAI_COMPATIBLE):
        module = importlib.import_module(f"quackd.agent.providers.{name}")
        vendor = getattr(module, OPENAI_COMPATIBLE[name])
        for m in models_for(name):
            assert vendor(model=m.id, client=FakeOpenAI()).supports_vision is m.vision, m.id


@pytest.mark.parametrize("name", sorted(OPENAI_COMPATIBLE))
def test_each_openai_shaped_vendor_is_wired_the_same_way(name: str) -> None:
    module = importlib.import_module(f"quackd.agent.providers.{name}")
    vendor = getattr(module, OPENAI_COMPATIBLE[name])
    assert vendor.name == name
    assert vendor.key_env == KEY_ENV[name]
    assert vendor.extra == EXTRA_FOR[name]
    assert vendor.base_url and vendor.base_url.startswith("https://"), (
        "a key goes to this address, so it is not going over plain http"
    )
    assert vendor.default_tool_choice in (None, "auto", "required", "any")
    assert vendor(client=FakeOpenAI()).model == default_model_for(name)


@pytest.mark.parametrize("name", sorted(OPENAI_COMPATIBLE))
def test_each_openai_shaped_vendor_names_its_own_key(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for env in set(KEY_ENV.values()) | {"CO_API_KEY", "MODEL_API_KEY"}:
        monkeypatch.delenv(env, raising=False)
    module = importlib.import_module(f"quackd.agent.providers.{name}")
    vendor = getattr(module, OPENAI_COMPATIBLE[name])
    with pytest.raises(ProviderMissingKey, match=KEY_ENV[name]):
        vendor()


@pytest.mark.parametrize(
    ("name", "fallback"), [("cohere", "CO_API_KEY"), ("meta", "MODEL_API_KEY")]
)
def test_the_two_vendors_with_a_second_key_name_accept_it(
    name: str, fallback: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both vendors' own examples export a different variable than their docs head with, and a
    reader who has one exported should not be told they have none."""
    monkeypatch.delenv(KEY_ENV[name], raising=False)
    monkeypatch.setenv(fallback, "sk-test")
    module = importlib.import_module(f"quackd.agent.providers.{name}")
    getattr(module, OPENAI_COMPATIBLE[name])(client=FakeOpenAI())


def test_the_catalogue_costs_nothing_to_import() -> None:
    """`quackd --help` and every press of TAB import this module, so it may not drag in pydantic
    or an SDK. Read as source rather than by importing, because by the time this test runs the
    suite has imported half of quackd already."""
    source = (REPO / "quackd" / "agent" / "providers" / "catalogue.py").read_text(encoding="utf-8")
    for forbidden in ("import pydantic", "from pydantic", "import openai", "import anthropic"):
        assert forbidden not in source


def test_every_vendor_file_is_reachable_from_the_factory() -> None:
    """A provider module nobody dispatches to is a vendor that silently does not exist."""
    directory = REPO / "quackd" / "agent" / "providers"
    modules = {p.stem for p in directory.glob("*.py")}
    plumbing = {"__init__", "base", "catalogue", "factory", "fake", "local", "openai"}
    assert modules - plumbing - {"anthropic", "gemini"} == set(OPENAI_COMPATIBLE)


def test_the_local_presets_are_the_ones_the_catalogue_excuses() -> None:
    """The catalogue and `local.py` keep separate lists, and they must name the same servers."""
    assert set(cat.LOCAL_NAMES) == set(PRESETS)
    for name in cat.LOCAL_NAMES:
        assert models_for(name) == ()
