"""Which decision LLM a run gets, and what reaches it.

`test_decision.py` next door is about the stepper: which turns are a choice, what an answer
has to clear, what the record says. This file is about everything in front of that -- the
flags, the variables, the table of presets, the two backends and the plugin hook -- because
that is the half that grew when Jev stopped being the only one.

The one rule worth stating up front: nothing in this file may need `typesafe_sdk` or `laya` to
be installed. Both are optional extras, CI installs neither, and a suite that quietly skipped
here would be a suite that never tested the thing this release is.
"""

from __future__ import annotations

import ast
import importlib.metadata
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

from quackd.agent.decision import factory
from quackd.agent.decision.base import DecisionError
from quackd.agent.decision.catalogue import (
    ENTRY_POINT_GROUP,
    IN_PROCESS,
    PRESET_NAMES,
    PRESETS,
    SYSTEM_ONE,
)
from quackd.agent.decision.factory import (
    decision_llm_is_available,
    find_preset,
    make_decision_llm,
    parse_decision_llm,
    preset_names,
    resolve_decision_mode,
    resolve_decision_model,
    resolve_decision_price,
    resolve_decision_url,
)
from tests import fake_laya, fake_systemone

ROOT = Path(__file__).resolve().parents[1]


# ── the table ───────────────────────────────────────────────────────────────────────────


def test_every_preset_row_is_complete() -> None:
    """A row is the whole definition of a decision LLM, so a half-filled one is a preset that
    names itself in `--help` and then fails somewhere further in."""
    for name, spec in PRESETS.items():
        assert spec.name == name, f"{name}: the row and its key disagree"
        assert spec.backend in (SYSTEM_ONE, IN_PROCESS), f"{name}: unknown backend"
        assert spec.summary and not spec.summary.endswith("."), f"{name}: summary is a label"
        assert spec.install, f"{name}: nothing says how to get it running"
        assert spec.sdk, f"{name}: no import to probe, so doctor cannot say if it is here"


def test_only_the_hosted_row_carries_a_key_a_price_or_a_model_variable() -> None:
    """The three things that belong to a vendor rather than to the format. A second row growing
    any of them is the table drifting back towards one-module-per-vendor."""
    for name, spec in PRESETS.items():
        hosted = name == "jev"
        assert bool(spec.key_env) is hosted, f"{name}: key_env"
        assert bool(spec.price) is hosted, f"{name}: price"
        assert bool(spec.model_env) is hosted, f"{name}: model_env"


def test_a_preset_url_carries_no_path() -> None:
    """The SDK appends `/v1/systemone` to whatever it is given, without normalising, so a row
    written in the style of `providers/local.py` -- which stores `http://host:8000/v1` --
    would ask for `/v1/v1/systemone` and get a 404 that named nothing."""
    for name, spec in PRESETS.items():
        if spec.url is None:
            continue
        assert spec.url.startswith("http://") or spec.url.startswith("https://"), name
        rest = spec.url.split("://", 1)[1]
        assert "/" not in rest, f"{name}: {spec.url} has a path; the SDK adds its own"


def test_only_the_two_rows_that_are_told_an_address_lack_one() -> None:
    """`local` exists to be told and `laya` has none to tell, and everything else listens
    somewhere quackd can name without being asked."""
    assert {name for name, spec in PRESETS.items() if spec.url is None} == {"jev", "local", "laya"}


def test_the_hosted_model_is_pinned_and_the_open_ones_are_named() -> None:
    """A moving alias is a run whose transcript describes a model that is no longer the one
    that answered. And the open servers do not share an id: OpenJev refuses a pinned Jev
    version with a 400, so a shared default would have been wrong there on every request."""
    assert PRESETS["jev"].model == "jev-1.13.0"
    assert PRESETS["openjev"].model != PRESETS["jev"].model
    assert PRESETS["local"].model is None


def test_the_extras_the_table_names_exist_and_install_what_it_says() -> None:
    """A row pointing at an extra nobody declared is an error message telling somebody to run
    an install that does nothing."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    for name, spec in PRESETS.items():
        assert spec.extra in extras, f"{name}: quackd[{spec.extra}] is not declared"
        installs = " ".join(extras[spec.extra])
        assert spec.sdk is not None
        assert spec.sdk.replace("_", "-") in installs, f"{name}: {spec.extra} does not install it"


def test_neither_optional_extra_is_in_all() -> None:
    """`quackd[all]` is the three pilot SDKs. A decision LLM generates nothing and cannot pilot
    a robot, and the in-process one would put torch in there."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    everything = " ".join(pyproject["project"]["optional-dependencies"]["all"])
    assert "typesafe" not in everything and "laya" not in everything


def test_the_catalogue_costs_nothing_to_import() -> None:
    """`--help` imports this table and so does every press of TAB. Nothing in it may reach for
    a vendor SDK, for torch, or for pydantic.

    Read off the import statements rather than off the text, because the module's own docstring
    names every one of these in the course of promising not to import them."""
    source = (ROOT / "quackd" / "agent" / "decision" / "catalogue.py").read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for heavy in ("typesafe_sdk", "laya", "torch", "pydantic", "httpx", "importlib"):
        assert heavy not in imported, f"catalogue.py imports {heavy}"


def test_importing_the_package_imports_no_backend() -> None:
    """The loop imports this on every run, including the overwhelming majority that name no
    decision LLM at all."""
    assert "typesafe_sdk" not in sys.modules
    assert "laya" not in sys.modules


# ── the spec ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("spec", "name", "model"),
    [
        ("jev", "jev", None),
        ("jev:jev-latest", "jev", "jev-latest"),
        ("KEV", "kev", None),
        ("  kev  ", "kev", None),
        ("laya:multilingual", "laya", "multilingual"),
        ("local:my-own-build", "local", "my-own-build"),
    ],
)
def test_a_spec_is_a_name_and_maybe_a_model(spec: str, name: str, model: str | None) -> None:
    parsed = parse_decision_llm(spec)
    assert parsed is not None
    found, asked = parsed
    assert (found.name, asked) == (name, model)


@pytest.mark.parametrize("spec", ["off", "OFF", "", "   "])
def test_off_and_blank_are_off(spec: str) -> None:
    """Off is the answer rather than a fallback: nothing here is switched on by being found."""
    assert parse_decision_llm(spec) is None


def test_nothing_named_anywhere_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(factory.ENV_LLM, "")
    assert parse_decision_llm(None) is None


def test_the_variable_names_one_when_the_flag_does_not(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(factory.ENV_LLM, "kev")
    parsed = parse_decision_llm(None)
    assert parsed is not None and parsed[0].name == "kev"


def test_the_flag_beats_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(factory.ENV_LLM, "kev")
    parsed = parse_decision_llm("jev")
    assert parsed is not None and parsed[0].name == "jev"


def test_the_flag_can_turn_off_what_the_variable_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A variable in a `.env` three directories up should be arguable with on the command
    line, and `--decision-llm off` is how you argue."""
    monkeypatch.setenv(factory.ENV_LLM, "jev")
    assert parse_decision_llm("off") is None


def test_a_name_nobody_defined_says_so_and_lists_the_ones_that_are(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(DecisionError) as e:
        parse_decision_llm("nosuch")
    message = str(e.value)
    assert "nosuch" in message and "--decision-llm" in message
    for name in PRESET_NAMES:
        assert name in message


def test_the_refusal_names_the_variable_when_the_variable_is_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flag somebody has just typed and a line in a `.env` they have forgotten want
    different sentences, and the second is the one that is hard to find."""
    monkeypatch.setenv(factory.ENV_LLM, "nosuch")
    with pytest.raises(DecisionError, match=factory.ENV_LLM):
        parse_decision_llm(None)


def test_a_model_keeps_its_own_colons() -> None:
    """Split at the first colon only. A checkpoint named like an Ollama tag is still one
    name, and cutting it in half would ask a server for something it does not serve."""
    parsed = parse_decision_llm("local:org/model:v2")
    assert parsed is not None and parsed[1] == "org/model:v2"


def test_a_trailing_colon_is_the_default_model() -> None:
    """Shell completion offers `kev:` as a prefix, and a reader may well press enter on it."""
    parsed = parse_decision_llm("kev:")
    assert parsed is not None and parsed[1] is None


# ── the mode ────────────────────────────────────────────────────────────────────────────


def test_naming_one_is_asking_for_it() -> None:
    """The default once you have named a decision LLM is `on`, not a second flag you also have
    to remember."""
    assert resolve_decision_mode(None, named=True) == "on"
    assert resolve_decision_mode(None, named=False) == "off"


@pytest.mark.parametrize("mode", ["off", "shadow", "on", "SHADOW"])
def test_a_mode_is_taken_as_written(mode: str) -> None:
    assert resolve_decision_mode(mode, named=True) == mode.lower()


def test_the_mode_variable_is_read_and_the_flag_beats_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(factory.ENV_MODE, "shadow")
    assert resolve_decision_mode(None, named=True) == "shadow"
    assert resolve_decision_mode("on", named=True) == "on"


def test_off_wins_over_a_named_decision_llm() -> None:
    """The mode is the switch, so it is the thing that can say no."""
    assert resolve_decision_mode("off", named=True) == "off"


def test_a_mode_nobody_defined_stops_the_run() -> None:
    with pytest.raises(DecisionError, match="maybe"):
        resolve_decision_mode("maybe", named=True)


def test_a_mode_with_nothing_to_run_it_says_which_flag_is_missing() -> None:
    """Silently doing nothing is the one outcome worth refusing here: a script that asked for
    shadow and got a plain run would go on producing plain runs indefinitely."""
    with pytest.raises(DecisionError) as e:
        resolve_decision_mode("shadow", named=False)
    assert "--decision-llm" in str(e.value) and "--decision-mode" in str(e.value)


def test_a_mode_from_a_variable_with_nothing_to_run_it_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(factory.ENV_MODE, "on")
    with pytest.raises(DecisionError, match=factory.ENV_MODE):
        resolve_decision_mode(None, named=False)


# ── the address, the model and the rate ─────────────────────────────────────────────────


def test_the_address_falls_back_from_flag_to_variable_to_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kev = PRESETS["kev"]
    assert resolve_decision_url(kev, "http://gpu:9000") == "http://gpu:9000"
    monkeypatch.setenv(factory.ENV_URL, "http://box:8009")
    assert resolve_decision_url(kev, None) == "http://box:8009"
    assert resolve_decision_url(kev, "http://gpu:9000") == "http://gpu:9000"
    monkeypatch.setenv(factory.ENV_URL, "")
    assert resolve_decision_url(kev, None) == kev.url


def test_the_hosted_row_has_no_address_of_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """Its address belongs to the SDK, which has its own variable for moving it."""
    monkeypatch.setenv(factory.ENV_URL, "")
    assert resolve_decision_url(PRESETS["jev"], None) is None


def test_the_model_variable_moves_the_hosted_row_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`TYPESAFE_DEFAULT_MODEL` is the SDK's own, so it is honoured exactly where the SDK owns
    the address too. A stray one must not retarget a server on somebody's own machine."""
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-preview")
    assert resolve_decision_model(PRESETS["jev"], None) == "jev-preview"
    assert resolve_decision_model(PRESETS["kev"], None) == PRESETS["kev"].model


def test_a_named_model_beats_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "jev-preview")
    assert resolve_decision_model(PRESETS["jev"], "jev-1.13.0") == "jev-1.13.0"


def test_a_server_you_run_is_free_and_the_hosted_one_is_not() -> None:
    """Not `None`, which quackd reserves for "there is a rate and nobody here knows it". A
    model on your own machine bills you in electricity, and that is the same `$0` the rest of
    quackd already prices a local pilot at."""
    assert resolve_decision_price(PRESETS["jev"]).input == 0.042
    for name in ("kev", "von", "openjev", "opendecision", "local", "laya"):
        price = resolve_decision_price(PRESETS[name])
        assert (price.input, price.output) == (0.0, 0.0), name
        assert price.source == "self-hosted", name


def test_the_price_variable_overrides_any_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """For the paid endpoint sitting behind `local`, which quackd has no way to know about."""
    monkeypatch.setenv(factory.PRICE_ENV, "in=1.5,out=0")
    price = resolve_decision_price(PRESETS["local"])
    assert price.input == 1.5 and price.source == factory.PRICE_ENV


# ── is it here at all ───────────────────────────────────────────────────────────────────


def test_the_client_being_absent_names_the_extra_that_installs_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    ok, why = decision_llm_is_available(PRESETS["kev"])
    assert not ok and "quackd[decision]" in why


def test_the_in_process_one_names_its_own_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "laya", None)
    ok, why = decision_llm_is_available(PRESETS["laya"])
    assert not ok and "quackd[laya]" in why


def test_the_hosted_one_needs_a_key_and_says_which(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_systemone.install(monkeypatch, fake_systemone.FakeDecisionLLM())
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    ok, why = decision_llm_is_available(PRESETS["jev"])
    assert not ok and "TYPESAFE_API_KEY" in why


def test_a_server_you_run_needs_no_key_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate that used to be "off when the key is missing" would otherwise keep every one
    of these off forever, because none of them has a key to miss."""
    fake_systemone.install(monkeypatch, fake_systemone.FakeDecisionLLM())
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    for name in ("kev", "von", "openjev", "opendecision", "local"):
        ok, why = decision_llm_is_available(PRESETS[name])
        assert ok, f"{name}: {why}"


def test_a_hosted_key_in_the_environment_does_not_switch_anything_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Somebody else's project left a key in a `.env`. That is not a request."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-not-a-request")
    monkeypatch.setenv(factory.ENV_LLM, "")
    assert parse_decision_llm(None) is None
    assert resolve_decision_mode(None, named=False) == "off"


# ── building one ────────────────────────────────────────────────────────────────────────


def test_a_server_you_run_gets_a_word_for_a_key_and_its_own_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hosted key must never be sent to a server on somebody's own machine, and the row's
    port must arrive without the SDK having to guess."""
    fake = fake_systemone.install(monkeypatch, fake_systemone.FakeDecisionLLM())
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-hosted-and-private")
    monkeypatch.setenv(factory.ENV_URL, "")
    llm = make_decision_llm(PRESETS["kev"])
    assert fake.api_key == "local"
    assert fake.base_url == "http://127.0.0.1:8009"
    assert llm.model == "kev-latest" and llm.name == "kev"


def test_the_hosted_one_is_handed_no_key_and_no_address(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both belong to the SDK, which reads its own variables for them and validates the key
    better than quackd could."""
    fake = fake_systemone.install(monkeypatch, fake_systemone.FakeDecisionLLM())
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-hosted")
    monkeypatch.setenv(factory.ENV_URL, "")
    make_decision_llm(PRESETS["jev"])
    assert fake.api_key is None and fake.base_url is None


def test_the_address_reaches_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = fake_systemone.install(monkeypatch, fake_systemone.FakeDecisionLLM())
    llm = make_decision_llm(PRESETS["local"], url="http://gpu.local:9001", model="my-build")
    assert fake.base_url == "http://gpu.local:9001"
    assert llm.url == "http://gpu.local:9001" and llm.model == "my-build"


def test_the_row_that_exists_to_be_told_an_address_refuses_without_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refused rather than guessed, the way `--llm local` refuses, and the refusal names both
    ways of saying it."""
    fake_systemone.install(monkeypatch, fake_systemone.FakeDecisionLLM())
    monkeypatch.setenv(factory.ENV_URL, "")
    with pytest.raises(DecisionError) as e:
        make_decision_llm(PRESETS["local"])
    assert "--decision-url" in str(e.value) and factory.ENV_URL in str(e.value)


def test_the_client_being_absent_is_a_sentence_rather_than_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    with pytest.raises(DecisionError, match=r"quackd\[decision\]"):
        make_decision_llm(PRESETS["kev"], url="http://127.0.0.1:8009")


# ── the one that runs in this process ───────────────────────────────────────────────────


async def test_the_in_process_one_answers_from_its_own_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = fake_laya.install(
        monkeypatch, fake_laya.FakeRouter(answers=fake_laya.turn("report_state"))
    )
    llm = make_decision_llm(PRESETS["laya"])
    result = await llm.decide({"goal": "look"}, {"next_verb": {"type": "choice"}})
    assert result["answers"]["next_verb"]["choice"] == "report_state"
    assert router.calls[0][0] == {"goal": "look"}


async def test_the_in_process_one_loads_once_and_asks_for_the_decision_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loading costs seconds against a timeout of one, so a second load would be a turn lost
    every turn. And the checkpoint asked for is the decision-tuned one: its plain name is an
    alias for the English model, which is a different thing that would answer anyway."""
    router = fake_laya.install(monkeypatch, fake_laya.FakeRouter(answers=fake_laya.turn("stop")))
    llm = make_decision_llm(PRESETS["laya"])
    await llm.decide({"goal": "a"}, {})
    await llm.decide({"goal": "b"}, {})
    assert router.loads == 1
    assert router.kwargs.get("preload") is True
    assert {model for _s, _q, model in router.calls} == {"typed-decisions"}


async def test_an_older_in_process_package_still_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The keyword that preloads the right checkpoint is younger than the class it belongs to,
    and quackd would rather load the wrong one once than refuse to run."""
    router = fake_laya.install(
        monkeypatch,
        fake_laya.FakeRouter(answers=fake_laya.turn("stop"), rejects=("default",)),
    )
    llm = make_decision_llm(PRESETS["laya"])
    await llm.decide({"goal": "a"}, {})
    # The construction that named the keyword raised before it built anything, so exactly one
    # router exists, and it is the one built without it.
    assert router.loads == 1
    assert router.kwargs.get("preload") is True
    assert "default" not in router.kwargs and router.kwargs.get("max_loaded") == 2


async def test_a_load_that_failed_is_not_retried_every_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken install should cost one attempt, not one per turn of a twelve-step run."""

    class Boom(RuntimeError):
        pass

    router = fake_laya.FakeRouter()

    def explode(**_kwargs: Any) -> Any:
        router.loads += 1
        raise Boom("no weights here")

    module = fake_laya.install(monkeypatch, router)
    monkeypatch.setattr(sys.modules["laya"], "Router", explode)
    llm = make_decision_llm(PRESETS["laya"])
    for _ in range(3):
        with pytest.raises(Boom):
            await llm.decide({"goal": "a"}, {})
    assert router.loads == 1
    assert module is router


def test_the_in_process_one_needs_no_address(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_laya.install(monkeypatch, fake_laya.FakeRouter())
    assert make_decision_llm(PRESETS["laya"]).url is None


# ── somebody else's ─────────────────────────────────────────────────────────────────────


class _Entry:
    """An `importlib.metadata.EntryPoint`, as far as the loader reads one."""

    def __init__(self, name: str, value: str) -> None:
        self.name, self.value, self.group = name, value, ENTRY_POINT_GROUP


@pytest.fixture
def plugin(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A third party's decision LLM, announced the only way one can be."""
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **kw: (
            [_Entry("stub", "tests.stub_decision_llm")]
            if kw.get("group") == ENTRY_POINT_GROUP
            else []
        ),
    )
    factory._plugins.cache_clear()
    yield sys.modules.get("tests.stub_decision_llm")
    factory._plugins.cache_clear()


def test_a_plugin_is_found_and_named(plugin: Any) -> None:
    assert "stub" in preset_names()
    spec = find_preset("stub")
    assert spec.name == "stub" and spec.summary == "a decision LLM that exists only in this test"
    assert spec.sdk is None  # it was found through its own metadata, so it is installed


async def test_a_plugin_is_built_and_asked(plugin: Any) -> None:
    import tests.stub_decision_llm as stub

    stub.built.clear()
    llm = make_decision_llm(find_preset("stub"), url="http://elsewhere:1234")
    assert stub.built and stub.built[-1] is llm
    assert llm.url == "http://elsewhere:1234" and llm.model == "stub-1"
    await llm.decide({"goal": "x"}, {"next_verb": {"type": "choice"}})
    assert llm.calls[0][0] == {"goal": "x"}


def test_a_plugin_is_named_in_a_refusal(plugin: Any) -> None:
    with pytest.raises(DecisionError, match="stub"):
        parse_decision_llm("nosuch")


def test_a_plugin_cannot_take_a_name_quackd_ships(monkeypatch: pytest.MonkeyPatch) -> None:
    """Otherwise a package on an index could quietly become the thing `--decision-llm jev`
    reaches, which is the whole supply-chain question in one line."""
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **kw: [_Entry("jev", "tests.stub_decision_llm")],
    )
    factory._plugins.cache_clear()
    try:
        assert find_preset("jev") is PRESETS["jev"]
    finally:
        factory._plugins.cache_clear()


def test_an_entry_point_whose_module_is_gone_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """An editable install keeps the metadata it was built with, so a module moved or removed
    since is still announced. Believing it turns a missing plugin into an ImportError from
    somewhere far away."""
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **kw: [_Entry("ghost", "tests.no_such_module_at_all")],
    )
    factory._plugins.cache_clear()
    try:
        assert "ghost" not in preset_names()
    finally:
        factory._plugins.cache_clear()
