"""The manifest is the source of truth for what a robot can do; these tests pin its rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import quackd.adapters
from quackd.adapters.manifest import RobotManifest, VerbSpec, manifest_json_schema
from quackd.adapters.microduck import microduck_manifest
from quackd.verbs.registry import ManifestError, default_registry, registry_from_manifest


def _manifest(**over: Any) -> RobotManifest:
    base: dict[str, Any] = {
        "id": "bot-01",
        "vendor": "acme",
        "model": "bot",
        "embodiment": "wheeled",
        "mobility": "wheeled",
        "intents": ["twist", "sound"],
        "sensors": ["camera"],
        "verbs": [VerbSpec(name="observe", core=True), VerbSpec(name="move", core=True)],
    }
    base.update(over)
    return RobotManifest(**base)


def test_round_trips_through_json() -> None:
    m = microduck_manifest("sim2d")
    assert RobotManifest.model_validate_json(m.model_dump_json()) == m


def test_stop_is_always_present_and_never_gated() -> None:
    m = _manifest()
    stop = m.verb("stop")
    assert stop is not None and stop.core and stop.safety_class == "safe"
    with pytest.raises(ValidationError, match="never be gated"):
        _manifest(verbs=[VerbSpec(name="stop", safety_class="confirm")])


def test_alias_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="not its alias"):
        _manifest(verbs=[VerbSpec(name="walk")])


def test_core_requirements_are_checked_on_the_manifest() -> None:
    with pytest.raises(ValidationError, match="needs mobility"):
        _manifest(mobility="none", verbs=[VerbSpec(name="go_to", core=True)])
    with pytest.raises(ValidationError, match="needs a camera"):
        _manifest(sensors=[], verbs=[VerbSpec(name="observe", core=True)])
    with pytest.raises(ValidationError, match="needs the sound intent"):
        _manifest(intents=["twist"], verbs=[VerbSpec(name="say", core=True)])
    with pytest.raises(ValidationError, match="not a core verb"):
        _manifest(verbs=[VerbSpec(name="dance", core=True)])
    # an extension may be called anything; a precondition may not name an undeclared verb
    _manifest(verbs=[VerbSpec(name="dance")])
    with pytest.raises(ValidationError, match="undeclared verb"):
        _manifest(preconditions={"fly": ["standing"]})


def test_extra_keys_and_bad_ids_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _manifest(colour="red")
    with pytest.raises(ValidationError, match="id must match"):
        _manifest(id="Not A Slug")
    with pytest.raises(ValidationError, match="duplicates"):
        _manifest(intents=["twist", "twist"])


def test_digest_is_a_capability_fingerprint() -> None:
    a = microduck_manifest("sim2d", "duck-a")
    b = microduck_manifest("mock", "duck-b")
    assert a.digest() == b.digest() and len(a.digest()) == 16
    assert _manifest().digest() != a.digest()


def test_provides_is_alias_aware() -> None:
    m = microduck_manifest("sim2d")
    assert m.provides("walk") and m.provides("move") and not m.provides("fly")
    spec = m.verb("get_frame")
    assert spec is not None and spec.name == "observe" and spec.core


def test_microduck_manifest_matches_the_default_registry() -> None:
    assert set(microduck_manifest("sim2d").verb_names()) == set(default_registry().names())


def test_registry_from_manifest_only_builds_declared_verbs() -> None:
    wheeled = registry_from_manifest(_manifest(intents=["twist"]))
    assert wheeled.names() == ["observe", "move", "stop"]  # no say without sound
    legless = registry_from_manifest(
        _manifest(
            mobility="none",
            intents=["gaze"],
            verbs=[VerbSpec(name="observe", core=True), VerbSpec(name="search_scan", core=True)],
        )
    )
    assert legless.names() == ["observe", "search_scan", "stop"]
    assert "move" not in legless and "walk_to" not in legless


def test_registry_from_manifest_refuses_what_no_code_implements() -> None:
    with pytest.raises(ManifestError, match="no implementation"):
        registry_from_manifest(_manifest(verbs=[VerbSpec(name="dance")]))
    with pytest.raises(ManifestError, match="no predicate"):
        registry_from_manifest(_manifest(preconditions={"move": ["upright"]}))


def test_manifest_schema_on_disk_is_current() -> None:
    on_disk = json.loads(
        (Path(quackd.adapters.__file__).with_name("manifest.schema.json")).read_text(
            encoding="utf-8"
        )
    )
    assert on_disk == manifest_json_schema(), "run: uv run python -m quackd.adapters.export"
