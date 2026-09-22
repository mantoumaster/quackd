"""Docs that make promises about code are checked against the code."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from quackd.agent.decision.catalogue import PRESET_NAMES, PRESETS
from quackd.agent.providers.factory import CLOUD_NAMES, KEY_ENV, default_model_for
from quackd.verbs.registry import default_registry
from quackd_microduck import upstream_api as up
from tests.adapter_layout import adapter_module

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text(encoding="utf-8")


def test_adapter_status_lists_every_microduck_upstream_ref() -> None:
    """The Microduck has two upstreams and adapter-status.md carries both: `robotd`'s API,
    and `microduck_rl`'s model and policies, which the physics backend fetches and runs.

    Every other upstream in the project has a doc-completeness guard, the six adapter pages
    through `test_adapter_doc_lists_every_upstream_ref` and `robotd` through this one. Without
    the second loop the newest table is the only one that can go stale in silence.
    """
    from quackd_microduck.sim3d import upstream_api as microduck_rl

    doc = (REPO / "docs" / "adapter-status.md").read_text(encoding="utf-8")
    missing = [ref.name for ref in up.all_refs() if ref.name not in doc]
    assert not missing, f"docs/adapter-status.md is missing: {missing}"
    unverified = [
        ref.name for ref in microduck_rl.refs_by_status("UNVERIFIED") if ref.name not in doc
    ]
    assert not unverified, f"adapter-status.md is missing microduck_rl assumptions: {unverified}"
    from quackd.adapters.factory import BACKENDS

    for adapter, backends in BACKENDS.items():
        for backend in backends:
            assert f"`{adapter}:{backend}`" in doc, f"adapter-status.md lacks {adapter}:{backend}"


def test_adapter_guide_and_manifest_spec_match_the_code() -> None:
    from quackd.adapters.factory import ADAPTER_NAMES
    from quackd.verbs.core import REQUIREMENTS

    guide = (REPO / "docs" / "adapters.md").read_text(encoding="utf-8")
    for name in ADAPTER_NAMES:
        assert f"`{name}`" in guide, f"docs/adapters.md does not mention {name}"
    for fn in ("describe", "implementations", "conditions", "make"):
        assert f"def {fn}(" in guide
    spec = (REPO / "docs" / "manifest-spec.md").read_text(encoding="utf-8")
    for verb in REQUIREMENTS:
        assert f"`{verb}`" in spec, f"docs/manifest-spec.md does not list core verb {verb}"
    assert "manifest.schema.json" in spec and "digest()" in spec


@pytest.mark.parametrize(
    "adapter",
    ["lerobot", "rosbridge", "open_duck", "xlerobot", "alohamini", "toddlerbot"],
)
def test_adapter_doc_lists_every_upstream_ref(adapter: str) -> None:
    api = adapter_module(adapter, "upstream_api")
    doc = (REPO / "docs" / "adapters" / f"{adapter}.md").read_text(encoding="utf-8")
    missing = [ref.name for ref in api.all_refs() if ref.name not in doc]
    assert not missing, f"docs/adapters/{adapter}.md is missing: {missing}"
    assert api.PIN[:7] in doc and "never" in doc.lower()  # the honesty label


def test_readme_promises() -> None:
    for needle in (
        "not affiliated with or endorsed by Pollen Robotics",
        "claude mcp add quackd",
        "dr-eureka",
        "github.com/pollen-robotics/microduck_rl",
        "--goal",
        "--llm fake",
        "biped",
        "pronounced",
        "One CLI for all your robots",
        "flock-hello",
        "quackd flock create",
        "Non goals for now",
        "--llm ollama",
        "quackd list-models",
        "docs/local-llms.md",
        "| Local models (",
        "--flock",
        "flock-kick",
        "docs/flock.md",
        "--no-log",
        "QUACKD_LOG",
    ):
        assert needle in README, needle
    # every vendor, from the code rather than a list here, so a vendor cannot be added to quackd
    # and left out of the one table that tells anyone it exists
    for name in CLOUD_NAMES:
        assert f"--llm {name}" in README, f"the README never shows --llm {name}"
        assert f"`{KEY_ENV[name]}`" in README, f"the README never names {KEY_ENV[name]}"
    assert "quadruped" not in README.lower()
    for hype in ("revolutionary", "world's first", "fully autonomous", "swarm intelligence"):
        assert hype not in README.lower(), hype


def test_the_readme_defaults_row_names_every_catalogue_default() -> None:
    """A default that is written down twice is a default that goes stale in one of them.

    This cannot stop the row being wrong the day a vendor moves, but it can stop the row being
    wrong the day quackd itself moves, which is what happened to `gpt-5`, `grok-4` and
    `gemini-2.5-pro`: the code changed under a sentence nobody re-read."""
    row = next((line for line in README.splitlines() if line.startswith("| Model |")), None)
    assert row, "the README's Configuration table has no Model row"
    for name in CLOUD_NAMES:
        default = default_model_for(name)
        assert f"`{default}`" in row, f"the Model row does not name {name}'s default, {default}"


def test_every_provider_key_is_named_where_keys_are_configured() -> None:
    """A vendor whose key variable is only in the source is a vendor nobody can authenticate."""
    env_example = (REPO / ".env.example").read_text(encoding="utf-8")
    for name in CLOUD_NAMES:
        assert f"\n{KEY_ENV[name]}=" in env_example, f".env.example has no {KEY_ENV[name]} line"


def test_the_catalogue_is_documented_where_it_is_configured() -> None:
    """The same rule the log is held to, for the thing that now decides every run's model."""
    for path, needles in (
        ("README.md", ("quackd list-models", "catalogue")),
        ("docs/faq.md", ("quackd list-models", "catalogue")),
        (".env.example", ("QUACKD_LLM", "list-models")),
        ("docs/local-llms.md", ("catalogue",)),
        # the one place the answer is that there is no answer, which is worth saying out loud
        ("docs/mcp.md", ("selects no model", "QUACKD_LLM")),
    ):
        text = (REPO / path).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, f"{path} does not mention {needle!r}"


#: The one dash the README is allowed: the leading one of a blockquote attribution,
#: `> — Name, Month Year`. That dash is the convention for signing a quote, not punctuation
#: inside a sentence, which is what the rule below is actually about. Only the prefix is
#: exempt. The rest of the line is held to the same standard as any other prose.
ATTRIBUTION_PREFIX = "> — "


def test_readme_punctuation_style() -> None:
    """House style: no semicolons and no dashes used as punctuation (em/en dash, ' - ').

    Fenced code blocks are exempt (YAML lists, shell comments, JSON are what they are), and
    so is the leading dash of a blockquote attribution (see ATTRIBUTION_PREFIX)."""
    prose = re.sub(r"```.*?```", "", README, flags=re.S)
    for i, line in enumerate(prose.splitlines(), 1):
        checked = line[len(ATTRIBUTION_PREFIX) :] if line.startswith(ATTRIBUTION_PREFIX) else line
        assert ";" not in checked, f"README:{i}: semicolon"
        for dash in ("—", "–", " - "):  # noqa: RUF001  (em dash, en dash, spaced hyphen)
            assert dash not in checked, f"README:{i}: dash punctuation {dash!r}"


def test_readme_ends_with_license_section() -> None:
    prose = re.sub(r"```.*?```", "", README, flags=re.S)  # ignore headings inside code blocks
    headings = re.findall(r"^## (.+)$", prose, flags=re.M)
    assert headings[-1] == "License", headings
    # a blank line (<br>) before every section, for breathing room on GitHub
    assert prose.count("<br>\n\n## ") == len(headings), "every H2 needs a <br> before it"


def test_readme_images_are_absolute_and_exist() -> None:
    srcs = re.findall(r'<img[^>]+src="([^"]+)"', README) + re.findall(
        r"!\[[^\]]*\]\(([^)\s]+)", README
    )
    assert srcs, "README has no images"
    raw = "https://raw.githubusercontent.com/rokbenko/quackd/main/"
    for src in srcs:
        assert src.startswith("https://"), f"relative image breaks on PyPI: {src}"
        if src.startswith(raw):
            path = src[len(raw) :].split("?", 1)[0]  # ?v=N busts GitHub's image cache
            assert (REPO / path).exists(), f"missing asset {src}"


def test_readme_verbs_match_registry() -> None:
    for name in default_registry().names():
        assert f"`{name}`" in README, f"README does not mention verb {name}"


def _emitted_kinds(*modules: str) -> set[str]:
    """Every literal event kind those modules emit, read out of the source. A kind nobody
    documented is a kind nobody knows to look for, and only the code knows them all."""
    import ast

    kinds: set[str] = set()
    for name in modules:
        tree = ast.parse((REPO / "quackd" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            fn = node.func
            called = fn.attr if isinstance(fn, ast.Attribute) else None
            if called in ("emit", "write", "_emit", "_event") and isinstance(
                node.args[0], ast.Constant
            ):
                value = node.args[0].value
                if isinstance(value, str) and value.islower():
                    kinds.add(value)
    return kinds


def test_the_docs_describe_every_log_event_the_code_emits() -> None:
    """architecture.md is the one place that enumerates the transcript, so it is the one
    place this can go stale."""
    # the three modules that write a *run* transcript. The flock keeps its own `flock.jsonl`
    # (docs/flock.md) and the MCP server's two envelope kinds are documented in docs/mcp.md.
    emitted = _emitted_kinds("agent/loop.py", "safety.py", "log.py")
    doc = (REPO / "docs" / "architecture.md").read_text(encoding="utf-8")
    missing = [kind for kind in sorted(emitted) if f"`{kind}`" not in doc]
    assert not missing, f"docs/architecture.md does not describe: {missing}"


def test_the_mcp_doc_describes_the_envelope_the_server_puts_round_a_call() -> None:
    """A model reading a tool result sees the server's own kinds first and last. They belong
    in the page the model's operator reads, not only in the one about the run loop."""
    emitted = _emitted_kinds("mcp_server.py")
    doc = (REPO / "docs" / "mcp.md").read_text(encoding="utf-8")
    architecture = (REPO / "docs" / "architecture.md").read_text(encoding="utf-8")
    missing = [k for k in sorted(emitted) if f"`{k}`" not in doc and f"`{k}`" not in architecture]
    assert not missing, f"neither docs/mcp.md nor architecture.md describes: {missing}"


def test_the_docs_name_every_gate_the_code_can_fire() -> None:
    """A gate is the difference between a refusal you can act on and an `ok: false` you
    cannot, so every one of them has to be findable by name in the docs."""
    import ast

    gates: set[str] = set()
    for name in ("safety.py", "mcp_server.py"):
        tree = ast.parse((REPO / "quackd" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if (fn.attr if isinstance(fn, ast.Attribute) else None) not in ("emit", "_emit"):
                continue
            if not node.args or getattr(node.args[0], "value", None) != "gate":
                continue
            for kw in node.keywords:
                if kw.arg == "gate" and isinstance(kw.value, ast.Constant):
                    gates.add(str(kw.value.value))
    assert gates, "no gate names found: the reader stopped seeing what the code emits"
    docs = "".join(
        (REPO / "docs" / name).read_text(encoding="utf-8") for name in ("architecture.md", "mcp.md")
    )
    missing = sorted(g for g in gates if f"`{g}`" not in docs)
    assert not missing, f"architecture.md and mcp.md name no gate called: {missing}"


def test_extra_body_is_documented_where_it_is_configured() -> None:
    """A knob nobody can find is a knob nobody has. This one is worse than most to discover by
    reading the source, because the field it carries belongs to the server rather than to
    quackd, so the name to search for is never in this repository at all."""
    for path, needles in (
        ("README.md", ("--extra-body", "QUACKD_EXTRA_BODY")),
        (
            "docs/local-llms.md",
            (
                "--extra-body",
                "QUACKD_EXTRA_BODY",
                "chat_template_kwargs",
                # the serve-time way round, so the docs do not imply the client is the only one
                "--default-chat-template-kwargs",
            ),
        ),
        ("docs/faq.md", ("--extra-body",)),
        (".env.example", ("QUACKD_EXTRA_BODY", "chat_template_kwargs")),
        # the page has no such door, and its own list of differences is where that is recorded
        ("web/README.md", ("extra_body",)),
    ):
        text = (REPO / path).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, f"{path} does not mention {needle!r}"


def test_the_log_is_documented_where_it_is_configured() -> None:
    for path, needles in (
        (
            "docs/architecture.md",
            (
                "## Log",
                "--no-log",
                "QUACKD_LOG",
                "QUACKD_LOG_PROMPT",
                "QUACKD_LOG_THINKING",
            ),
        ),
        ("docs/mcp.md", ("log", "--no-log", "QUACKD_LOG")),
        ("docs/safety.md", ("--dry-run", "dry_run")),
        (".env.example", ("QUACKD_LOG", "QUACKD_LOG_THINKING", "QUACKD_LOG_PROMPT")),
        ("docs/flock.md", ("log", "--no-log")),
    ):
        text = (REPO / path).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, f"{path} does not mention {needle!r}"


def test_the_price_of_a_run_is_documented_where_it_is_configured() -> None:
    """The same rule the log and the catalogue are held to, for the thing that decides
    whether a run's cost counter reads a number or `cost unpriced`.

    A rate is the one knob here somebody only goes looking for after a bill, so it has to be
    findable from the page they are already on: the README's usage table for the flag, the
    architecture page for the order the three sources are tried in, `.env.example` for the
    variable, and decision-llms.md for the stepper's own rate, which is a separate variable
    because it prices a separate vendor's tokens.

    The `.env.example` needles carry their `=` on purpose. `QUACKD_PRICE` is also spelled in
    the prose above the stepper's variable ("the same syntax as QUACKD_PRICE above"), so a
    bare substring check would still pass with the line that actually sets it deleted."""
    for path, needles in (
        ("README.md", ("--price", "--run-name")),
        ("docs/architecture.md", ("QUACKD_PRICE", "--price", "--run-name")),
        ("docs/decision-llms.md", ("QUACKD_DECISION_PRICE",)),
        (".env.example", ("QUACKD_PRICE=", "QUACKD_DECISION_PRICE=")),
        # where somebody lands who has already been surprised by a figure, or by its absence
        ("docs/faq.md", ("--price", "--run-name", "unpriced")),
    ):
        text = (REPO / path).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, f"{path} does not mention {needle!r}"


def test_the_decision_llm_is_documented_where_it_is_configured() -> None:
    """The same rule again, for the three flags and the three variables that decide whether a
    discrete stepper answers a turn before the model is ever called.

    There are many decision LLMs now rather than one, so the names a reader has to be able to
    find are not a vendor's: they are `--decision-llm` and `--decision-mode` on the front page,
    because that is where somebody decides whether to read any further, and on the page itself
    the address for a server quackd has never heard of, the two extras that install the two
    ways of reaching one, and the entry point group a third party publishes into.

    `TYPESAFE_API_KEY=` carries its `=` because the key is also named in the prose around it,
    and a key nobody can set is a hosted model nobody can run."""
    for path, needles in (
        ("README.md", ("--decision-llm", "--decision-mode")),
        (
            "docs/decision-llms.md",
            ("--decision-url", "quackd[decision]", "quackd[laya]", "quackd.decision_llms"),
        ),
        (
            ".env.example",
            (
                "QUACKD_DECISION_LLM",
                "QUACKD_DECISION_MODE",
                "QUACKD_DECISION_URL",
                "TYPESAFE_API_KEY=",
            ),
        ),
    ):
        text = (REPO / path).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, f"{path} does not mention {needle!r}"


def test_task_pictures_are_documented_where_they_are_configured() -> None:
    """A flag whose whole point is that the model can see the thing has to be findable by
    somebody who has the thing and does not know the flag exists. The README quickstart is
    where an arm owner starts, so it is named there as well as in the reference pages."""
    for path, needles in (
        ("README.md", ("--image", "--vision")),
        ("docs/lerobot-first-run.md", ("--image", "--vision")),
        ("docs/local-llms.md", ("--image", "--vision")),
        ("docs/adapters/lerobot.md", ("--image",)),
        # the pictures land in the run directory, so the page that draws that directory says so
        ("docs/architecture.md", ("images/",)),
    ):
        text = (REPO / path).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, f"{path} does not mention {needle!r}"


def test_the_hand_placed_start_is_documented_where_it_is_configured() -> None:
    """The one place quackd takes torque off a robot. Somebody about to hold an arm while it is
    released should be able to find what happens next in the page they are already reading, and
    the safety page has to carry it whether or not they ever open the arm's own."""
    for path, needles in (
        ("README.md", ("--by-hand",)),
        ("docs/lerobot-first-run.md", ("--by-hand",)),
        ("docs/adapters/lerobot.md", ("--by-hand",)),
        ("docs/lerobot-hardware-checklist.md", ("--by-hand",)),
        ("docs/safety.md", ("--by-hand", "torque")),
    ):
        text = (REPO / path).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, f"{path} does not mention {needle!r}"


def test_both_ways_a_run_can_start_are_offered_together() -> None:
    """Rok asked for this specifically: the default and the hand-placed start are two options a
    reader chooses between, so the pages that teach the arm present them side by side rather
    than leaving the second one to a reference page nobody reaches.

    Checked as proximity rather than as wording, because the wording is prose and will be
    rewritten: what must survive a rewrite is that `--by-hand` is explained on the same page as
    the rest pose it departs from, and near it."""
    for path in ("README.md", "docs/lerobot-first-run.md"):
        lines = (REPO / path).read_text(encoding="utf-8").splitlines()
        rest = [i for i, line in enumerate(lines) if "rest-pose" in line or "rest pose" in line]
        hand = [i for i, line in enumerate(lines) if "--by-hand" in line]
        assert rest and hand, f"{path} must name both the rest pose and --by-hand"
        gap = min(abs(h - r) for h in hand for r in rest)
        assert gap < 60, (
            f"{path} explains --by-hand {gap} lines from the nearest mention of the rest pose; "
            "the two starts are a choice and belong next to each other"
        )


def test_mcp_doc_lists_every_tool() -> None:
    from quackd.mcp_server import TOOL_NAMES

    doc = (REPO / "docs" / "mcp.md").read_text(encoding="utf-8")
    missing = [name for name in TOOL_NAMES if f"`{name}" not in doc]
    assert not missing, f"docs/mcp.md is missing: {missing}"
    assert "--robots" in doc and "--robots" in README


def test_mcp_json_is_a_stdio_server() -> None:
    from quackd.adapters.factory import BACKENDS

    cfg = json.loads((REPO / ".mcp.json").read_text(encoding="utf-8"))
    server = cfg["mcpServers"]["quackd"]
    assert "command" in server and "type" not in server
    args = server["args"]
    assert "serve-mcp" in args
    # the robot it names has to exist, or opening the repo greets you with a stack trace
    adapter, _, backend = args[args.index("--robot") + 1].partition(":")
    assert backend in BACKENDS.get(adapter, ()), f".mcp.json names {adapter}:{backend}"
    # This is the repo's own config, so it runs the code you are editing, not the release.
    # `uv run` alone re-syncs on launch and loses to the running server's hold on
    # Scripts/quackd.exe on Windows, so the repo pins --no-sync. Users get `uvx` (docs/mcp.md).
    if server["command"] == "uv" and args[0] == "run":
        assert "--no-sync" in args, "uv run re-syncs and fights the server it is launching"


# ── counts, so a release cannot ship a number the code disagrees with ────────────────────

_NUMBER_WORDS = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}


def _prose(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def _living_docs() -> list[Path]:
    """Every document that describes quackd as it is now.

    CHANGELOG, PLAN and the ADRs and design notes are excluded: they record what was true
    when they were written, and correcting a number in them would be falsifying history."""
    return [
        path
        for path in sorted(REPO.glob("*.md")) + sorted((REPO / "docs").rglob("*.md"))
        if path.name not in ("CHANGELOG.md", "PLAN.md") and not {"design", "adr"} & set(path.parts)
    ]


@pytest.mark.parametrize("name", ["README.md", "docs/adapters.md", "docs/faq.md", "LAUNCH.md"])
def test_no_document_claims_the_wrong_number_of_adapters(name: str) -> None:
    """Half of the 0.5 documentation audit was stale counts that no test could see.

    "four adapters" was written in six places while five shipped. This does not police
    prose, only the specific claim that quackd has N adapters."""
    from quackd.adapters.factory import ADAPTER_NAMES

    right = _NUMBER_WORDS[len(ADAPTER_NAMES)]
    prose = _prose((REPO / name).read_text(encoding="utf-8")).lower()
    # only claim shapes that are unambiguously about how many adapters exist. "two robots
    # under one contract" is a heterogeneous flock, not a count of adapters.
    shapes = ("{w} adapters", "{w} robots supported", "{w} robots today")
    for count, word in _NUMBER_WORDS.items():
        if count == len(ADAPTER_NAMES):
            continue
        for shape in shapes:
            claim = shape.format(w=word)
            assert claim not in prose, (
                f"{name} says {claim!r}; quackd ships {right} ({', '.join(ADAPTER_NAMES)})"
            )


def test_no_living_document_claims_the_wrong_number_of_cloud_providers() -> None:
    """The same failure the adapter count already has a guard for, one layer up.

    "The four cloud providers see the camera frame as an image" was written once and was true
    for a year. Eleven is a number that will move again, and every document that spells it is a
    document nobody will re-read on the day it does."""
    right = _NUMBER_WORDS[len(CLOUD_NAMES)]
    shapes = ("{w} cloud providers", "{w} cloud vendors", "{w} vendors")
    # `_living_docs` stops at `docs/`, and the browser demo keeps its own count of the same
    # vendors in its README and in the header of the file that decides which it offers. Those
    # were both wrong the first time this list was written, which is the argument for including
    # them: a count is only checkable where somebody thought to look.
    web = [REPO / "web" / "README.md", REPO / "web" / "src" / "providers.js"]
    for path in [*_living_docs(), *web]:
        prose = _prose(path.read_text(encoding="utf-8")).lower()
        for count, word in _NUMBER_WORDS.items():
            if count == len(CLOUD_NAMES):
                continue
            for shape in shapes:
                claim = shape.format(w=word)
                assert claim not in prose, (
                    f"{path.name} says {claim!r}; quackd has {right} ({', '.join(CLOUD_NAMES)})"
                )


def test_the_pypi_summary_names_every_robot() -> None:
    """The one sentence on the PyPI page shipped 0.5 without the Open Duck Mini in it.

    That line and the keywords are how someone searching for their robot finds quackd, and
    nothing in the test suite had ever read pyproject.toml."""
    import tomllib

    from quackd.adapters.factory import ADAPTER_NAMES

    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    haystack = (project["description"] + " " + " ".join(project["keywords"])).lower()
    # the summary names bodies, not adapter identifiers: microduck -> "microduck",
    # open_duck -> "open duck", rosbridge -> "ros"
    for adapter in ADAPTER_NAMES:
        needle = {"open_duck": "open duck", "rosbridge": "ros"}.get(adapter, adapter)
        assert needle in haystack, (
            f"pyproject describes quackd without {needle!r}; it ships a {adapter} adapter"
        )


def test_the_readme_starter_table_lists_every_bundled_duck() -> None:
    """`open-duck-lookout` shipped in 0.5 and appeared nowhere in the README."""
    from quackd.duckfile.parser import list_bundled_ducks

    missing = [p.stem for p in list_bundled_ducks() if f"`{p.stem}`" not in README]
    assert not missing, f"README does not mention: {missing}"


def test_no_living_document_quotes_an_exact_test_count() -> None:
    """CONTRIBUTING said 445 while 451 ran, and no test could see it.

    Collecting the suite to check a number would cost every run eight seconds for a fact
    nobody reads, so the rule is simply not to quote one outside the history files, where
    a count is a record of what was true on the day and must not be rewritten."""
    for path in _living_docs():
        for claim in re.findall(r"\b\d{2,4} tests?\b", _prose(path.read_text(encoding="utf-8"))):
            pytest.fail(f"{path.name} quotes {claim!r}; say 'the whole suite' and let CI count")


def test_no_document_still_promises_a_removal_that_happened() -> None:
    """0.4 said `--transport` and the duck_* tools go in 0.5. They did, so nothing should
    still be promising it, and nothing should still be offering them.

    A promise about a release that has NOT happened is a different thing and is allowed: 0.11
    renamed the trace to the log and says the old spellings go in 0.12, which is exactly the
    shape of the promise 0.4 made and 0.5 kept. What this guards against is the stale half,
    a document still describing a removal that is already behind us."""
    from quackd.mcp_server import TOOL_NAMES

    assert not [n for n in TOOL_NAMES if n.startswith("duck_")]
    # a table title and a TransportError still told users to pass it, and no doc test could
    # see a Python string, so the same rule now covers the source that prints to a terminal
    for src in sorted((REPO / "quackd").rglob("*.py")):
        assert "--transport" not in src.read_text(encoding="utf-8"), (
            f"quackd/{src.relative_to(REPO / 'quackd').as_posix()} still offers --transport"
        )
    for path in _living_docs():
        text = _prose(path.read_text(encoding="utf-8"))
        assert "--transport" not in text, f"{path.name} still documents --transport"
        for promise in ("go away in 0.5", "gone in 0.5", "are removed in 0.5"):
            assert promise not in text, f"{path.name} still promises {promise!r}, which happened"


#: The one-liner ADR-0002 minted and ADR-0035 retired. ADR-0002 said "one-liner everywhere",
#: and everywhere is what happened: the README, the PyPI summary, the `quackd --help` banner,
#: the package docstring and the browser demo's title all carried a copy of it, and nothing
#: counted them. quackd is the CLI and the LLM is the brain now, so "an LLM for a brain" is
#: fine and only the old claim, that quackd itself is a brain for one small robot, is not.
_RETIRED_TAGLINES = (
    "Give your Microduck a brain",
    "Give your small robot a brain",
    "Give a Microduck a brain",
    "a brain for any small robot",
    "brain daemon Microduck was missing",
)

#: The three user-facing strings outside the README that carry the replacement, so a revert
#: in one of them cannot pass while the README still reads correctly.
_TAGLINE_SITES = ("pyproject.toml", "quackd/__init__.py", "quackd/cli.py")


def test_the_retired_tagline_is_gone_and_the_new_one_is_everywhere_it_lived() -> None:
    """The history files keep it: they record what was true on the day they were written."""
    # rglob, not glob: the retired sentence also lived in `quackd/agent/prompts.py`'s neighbours,
    # and a guard that reads only the top of the package is a guard with a floor under it
    sources = [
        REPO / "pyproject.toml",
        *(REPO / "quackd").rglob("*.py"),
        *(REPO / "bridge").rglob("*.py"),
        REPO / "web/index.html",
    ]
    for path in _living_docs() + sources:
        text = path.read_text(encoding="utf-8")
        for retired in _RETIRED_TAGLINES:
            assert retired not in text, f"{path.name} still carries the retired tagline {retired!r}"
    for name in _TAGLINE_SITES:
        text = (REPO / name).read_text(encoding="utf-8")
        assert "One CLI for all your robots" in text, f"{name} does not carry the one-liner"


#: Sentences that were true until 2026-09-15 and are not any more. An SO-101 arm ran quackd
#: that day, so a living document that still says nothing ever has is not merely stale, it is
#: telling a reader the opposite of what happened. The history files keep them on purpose:
#: CHANGELOG, PLAN and everything under docs/adr and docs/design record what was true when
#: they were written, which is what `_living_docs()` already excludes.
_RETIRED_HARDWARE_CLAIMS = (
    "no robot of any kind has run quackd",
    "has never run on any of them",
    "nothing in quackd has ever run on a real so-101",
    "never run on an arm",
    "nobody has pointed any pilot at a real webcam",
    "nothing has run on a real robot of any kind",
    "never run against an arm",
    # three more spellings, retired when the README started leading with the arm
    "nothing here has run on hardware",
    "nobody has run it on real hardware",
    "nobody has run `lerobot:real`",
)


def test_no_living_document_still_says_no_robot_has_ever_run_quackd() -> None:
    """One body has run on hardware and six have not, and both halves have to survive edits.

    The hard part of this change was not flipping the status, it was that the old claim was
    spelled seven different ways across a dozen pages, and a revert in any one of them reads
    as authoritative. The user-facing strings are checked too, because `quackd doctor` and
    `list-adapters` print a status line per adapter and those are read more often than a page.
    """
    sources = [
        REPO / "pyproject.toml",
        *(REPO / "quackd").rglob("*.py"),
        *(REPO / "adapters").rglob("*.py"),
        # the issue templates are read by the one person best placed to correct us, and the
        # arm's said "nobody has run this" for two days after somebody had
        *sorted((REPO / ".github" / "ISSUE_TEMPLATE").glob("*.yml")),
    ]
    for path in _living_docs() + sources:
        # compared in lower case: the same claim was capitalised differently per page, which
        # is how "Nothing here has run on hardware" outlived its lower case twin
        text = path.read_text(encoding="utf-8").lower()
        for retired in _RETIRED_HARDWARE_CLAIMS:
            assert retired.lower() not in text, (
                f"{path.relative_to(REPO)} still says {retired!r}; an SO-101 ran quackd on "
                "2026-09-15 (docs/adapter-status.md)"
            )
    # and the other half: the six that have not run must not be quietly promoted with it
    status = (REPO / "docs" / "adapter-status.md").read_text(encoding="utf-8")
    assert "2026-09-15" in status, "adapter-status.md does not date the one real run"


def test_the_readme_hero_is_the_real_arm_and_its_caption_says_when_and_who() -> None:
    """The front door's most checkable claim, and the one a well meaning edit would soften.

    Until 2026-09-15 the hero was a render of a duck walked by the scripted pilot, and every
    honest caption said so. It is a phone recording of a real arm under a real model now. The
    needles rather than the wording, so the prose stays free: which file, and that the caption
    dates it, names the model and names the body. And that it does not call itself scripted,
    which is the one word that would make it the old claim again."""
    gifs = re.findall(
        r'src="https://raw\.githubusercontent\.com/rokbenko/quackd/main/(docs/assets/[^"?]+\.gif)',
        README,
    )
    assert gifs, "the README shows no GIF at all"
    assert gifs[0] == "docs/assets/lerobot.gif", (
        f"the README's first GIF is {gifs[0]}. The hero is the recording of the real arm."
    )
    # anchored to the hero's own <p>: an unanchored search walks past a missing caption and
    # binds these needles to the next figure's, so deleting the caption would pass
    block = re.search(r"<p align=\"center\">(?:(?!</p>).)*?lerobot\.gif.*?</p>", README, flags=re.S)
    assert block is not None, 'the hero is not in a <p align="center"> block'
    caption = re.search(r"<sub>(.*?)</sub>", block.group(0), flags=re.S)
    assert caption is not None, "the hero has no caption"
    for needle in ("2026-09-15", "gpt-6-astra", "SO-101"):
        assert needle in caption.group(1), f"the hero caption does not say {needle}"
    assert "scripted" not in caption.group(1).lower(), "the hero is not the scripted pilot"


def test_every_file_in_docs_assets_has_a_row_in_its_catalogue() -> None:
    """`docs/assets/README.md` is where a reader finds out how a picture was made and whether
    a model or a script drove it. An asset with no row is a picture with no provenance, which
    for a recording of a real robot is the difference between evidence and decoration."""
    catalogue = (REPO / "docs" / "assets" / "README.md").read_text(encoding="utf-8")
    for asset in sorted((REPO / "docs" / "assets").iterdir()):
        if not asset.is_file() or asset.suffix == ".py" or asset.name == "README.md":
            continue
        assert f"| `{asset.name}` |" in catalogue, (
            f"docs/assets/{asset.name} has no row in docs/assets/README.md: say what it is "
            "and how it was made, and whether the pilot in it was a model or the script."
        )


def test_no_living_document_or_user_facing_string_still_says_fleet() -> None:
    """One word for a group of robots, because two words for one idea is two ideas to a reader.

    ADR-0035 retired "fleet" from prose and from help text, and kept it in the code, where
    `Fleet` and `build_fleet_server` are imported by name. The split is the point: this guard
    reads what a person reads, which for `quackd/` means docstrings and help strings rather
    than identifiers, so it checks the CLI's rendered help rather than the source."""
    from typer.testing import CliRunner

    from quackd.cli import app

    for path in _living_docs():
        prose = _prose(path.read_text(encoding="utf-8")).lower()
        assert "fleet" not in prose, f"{path.name} still says fleet"
    # the help a user actually sees, top level and every sub-app, which is where three of the
    # stale ones were: a source grep would have been satisfied by `fleet_from_flags`
    runner = CliRunner()
    for argv in ([], ["robot"], ["flock"], ["memory"], ["run"], ["serve-mcp"], ["validate"]):
        result = runner.invoke(app, [*argv, "--help"])
        # exit code first: a renamed sub-app makes Click print a "No such command" usage page,
        # which contains no "fleet" either, and the assertion below would pass on nothing
        assert result.exit_code == 0, f"`quackd {' '.join(argv)} --help` did not render"
        assert "fleet" not in result.output.lower(), (
            f"`quackd {' '.join(argv)} --help` still says fleet"
        )


def test_no_living_document_claims_the_wrong_number_of_mcp_tools() -> None:
    """The guard above proves nothing still *offers* a removed tool. It could not see a
    document still *describing* one, so architecture.md went through the whole of 0.5 saying
    the server carried the old count plus the duck_* aliases that release deleted, and into
    0.6, which added two more tools. Two README sentences, a --robots help string, a module
    docstring and a test docstring carried the old count for the same reason: nothing counted
    them. This file is scanned too, which is why the stale wordings are described here rather
    than quoted."""
    from quackd.mcp_server import TOOL_NAMES

    right = _NUMBER_WORDS[len(TOOL_NAMES)]
    # 0.5 learned that a doc-only guard cannot see a Python string a user reads: the count
    # was also stale in a `--robots` help text, a module docstring and a test's docstring
    # this file is skipped because it has to spell the wordings it forbids in order to
    # forbid them; every other source file and living document is fair game
    here = Path(__file__).resolve()
    sources = [
        p
        for p in sorted((REPO / "quackd").rglob("*.py")) + sorted((REPO / "tests").glob("*.py"))
        if p.resolve() != here
    ]
    for path in _living_docs() + sources:
        text = path.read_text(encoding="utf-8")
        haystack = (_prose(text) if path.suffix == ".md" else text).lower()
        for count, word in _NUMBER_WORDS.items():
            if count == len(TOOL_NAMES):
                continue
            for shape in (f"{word} `robot_*` tools", f"{word} robot_* tools"):
                assert shape not in haystack, (
                    f"{path.name} says {shape!r}; the server registers {right} "
                    f"({', '.join(TOOL_NAMES)})"
                )
        # anything that still describes the removed aliases as present, in any wording
        for stale in ("duck_* tools kept as aliases", "`duck_*` tools kept as aliases"):
            assert stale not in haystack, f"{path.name} describes the duck_* aliases as present"


# ── the guard that was missing twice ────────────────────────────────────────────────────


def test_the_architecture_diagram_names_every_adapter() -> None:
    """`_prose()` strips fenced blocks before every other doc guard, so the mermaid diagram
    is invisible to all of them by construction.

    That is not hypothetical. `docs/design/memory.md` records 0.6 fixing exactly this defect
    ("the README's architecture diagram listed four adapters and omitted `open_duck` and its
    `bridge` backend, which the adapter-count guard could not see because it reads the phrase
    'N adapters' and not a list"). Nothing was added to catch it, so it came back three
    adapters later. This is that guard.
    """
    from quackd.adapters.factory import ADAPTER_NAMES, BACKENDS

    node = next((line for line in README.splitlines() if 'ADAPTER["robot adapter' in line), None)
    assert node is not None, "the architecture diagram's adapter node has moved or gone"

    # The names as a SET, split on the separator, not as substrings. `"lerobot" in node` is
    # satisfied by the `xlerobot` entry, so a substring check cannot see `lerobot` go missing,
    # which is the one adapter whose name is contained in another's.
    listed = {n.strip() for n in node.split("<br/>")[1].split("·")}
    missing = [name for name in ADAPTER_NAMES if name not in listed]
    assert not missing, f"the architecture diagram does not name: {missing} (has {listed})"
    extra = [name for name in listed if name and name not in ADAPTER_NAMES]
    assert not extra, f"the architecture diagram names adapters that do not exist: {extra}"

    #: Backends are listed by their bare name in that node, so every distinct one must appear.
    kinds = {backend for backends in BACKENDS.values() for backend in backends}
    absent = sorted(k for k in kinds if k not in node)
    assert not absent, f"the architecture diagram does not name the backends: {absent}"


def test_no_fenced_block_names_a_stale_subset_of_the_adapters() -> None:
    """The general form of the same hole: any fenced block that enumerates most of the
    adapters has to enumerate all of them, or it is a list somebody forgot to update."""
    from quackd.adapters.factory import ADAPTER_NAMES

    for doc in [REPO / "README.md", *sorted((REPO / "docs").rglob("*.md"))]:
        text = doc.read_text(encoding="utf-8")
        for block in re.findall(r"```.*?```", text, flags=re.S):
            named = [n for n in ADAPTER_NAMES if n in block]
            if len(named) < len(ADAPTER_NAMES) - 2:
                continue  # not an enumeration, just a couple of examples
            missing = [n for n in ADAPTER_NAMES if n not in block]
            assert not missing, (
                f"{doc.relative_to(REPO)}: a fenced block names {len(named)} adapters "
                f"and omits {missing}"
            )


#: The README's verb table names bodies, not adapter ids, so the mapping is written down.
_VERB_TABLE_ROWS = {
    "microduck": "| Microduck |",
    "lerobot": "| LeRobot arm |",
    "rosbridge": "| rosbridge base |",
    "open_duck": "| Open Duck Mini v2 |",
    "xlerobot": "| XLeRobot |",
    "alohamini": "| AlohaMini |",
    "toddlerbot": "| ToddlerBot |",
}

_CORE_VERBS = frozenset(
    {"observe", "report_state", "stop", "say", "move", "go_to", "search_scan", "approach_and"}
)


async def _implementable(adapter: str) -> set[str]:
    """Every verb this adapter has an implementation for, across all its builds."""

    module = adapter_module(adapter)
    return set(module.implementations())


def test_the_readme_verb_table_has_a_row_per_body_listing_its_real_verbs() -> None:
    """The other list-shaped thing no guard could see.

    `test_readme_verbs_match_registry` only checks that each *core* verb appears somewhere in
    the whole README, so a body could be added with its own verbs and never get a row. Four
    were: Open Duck Mini, XLeRobot, AlohaMini and ToddlerBot all shipped verbs of their own
    with nothing in the table.
    """
    import asyncio

    from quackd.adapters.factory import ADAPTER_NAMES, make_adapter

    assert set(_VERB_TABLE_ROWS) == set(ADAPTER_NAMES), "the row map has drifted from the code"

    offline = {"microduck": "sim2d", "lerobot": "mock", "rosbridge": "mock"}
    for adapter in ADAPTER_NAMES:
        backend = offline.get(adapter, "mock")
        manifest = asyncio.run(make_adapter(f"{adapter}:{backend}", seed=0).connect())
        own = sorted(set(manifest.verb_names()) - _CORE_VERBS)
        prefix = _VERB_TABLE_ROWS[adapter]
        row = next((line for line in README.splitlines() if line.startswith(prefix)), None)
        assert row is not None, f"the verb table has no row for {adapter}"
        if not own:
            continue  # a body with nothing of its own says so in prose
        # The verbs cell only. Searching the whole row lets the description satisfy it, and
        # these descriptions name verbs: the first version of this guard passed happily with
        # two of the ToddlerBot's three verbs deleted from the cell.
        cell = row.split("|")[2]
        missing = [v for v in own if f"`{v}`" not in cell]
        assert not missing, f"the {adapter} row does not list its own verbs: {missing}"
        # And the other direction, which is the drift that happens when a verb is deleted from
        # an adapter and nobody remembers the README.
        listed = {chunk.strip() for chunk in cell.split("`") if chunk.strip()}
        # Against everything the adapter can implement, not just what this build reports: a
        # row may name a verb only some builds have (the ToddlerBot's `grip` needs the gripper
        # variant), but it may never name one the adapter cannot implement at all.
        possible = set(asyncio.run(_implementable(adapter)))
        gone = [v for v in listed if v not in possible and v not in _CORE_VERBS]
        assert not gone, f"the {adapter} row lists verbs the adapter cannot implement: {gone}"


def test_every_body_carries_its_own_numbers_on_its_own_page() -> None:
    """A datasheet is a claim about a real robot, so it belongs where a reader checks it.
    rosbridge is exempt: its numbers come off a bridge and are different every time."""
    from quackd.adapters.factory import ADAPTER_NAMES, BACKENDS, RobotSpec, describe

    pages = {
        "microduck": REPO / "docs" / "adapter-status.md",
        **{
            name: REPO / "docs" / "adapters" / f"{name}.md"
            for name in ADAPTER_NAMES
            if name not in ("microduck", "rosbridge")
        },
    }
    for adapter, path in pages.items():
        page = path.read_text(encoding="utf-8")
        sheet = describe(RobotSpec(adapter, BACKENDS[adapter][0])).datasheet
        assert sheet is not None
        for label, figure, unit in sheet.known():
            low = getattr(figure, "low", None)
            amount = (
                f"{low:g} to {figure.high:g} {unit}".rstrip()  # a band
                if low is not None
                else f"{figure.value:g} {unit}".rstrip()
            )
            assert amount in page, f"{path.name} does not say {label} is {amount}"


def test_the_registry_is_documented_where_it_is_configured() -> None:
    """Two files under a directory an env var moves, holding a robot's token. Every one of
    those facts has a place it has to be findable from, or somebody loses a robot or a secret.
    """
    for path, needles in (
        (
            "README.md",
            ("quackd robot", "quackd flock", "QUACKD_REGISTRY_DIR", "--registry-dir", "rest-pose"),
        ),
        (
            "docs/registry.md",
            ("robots.json", "flocks.json", "--probe", "plain text", "rest-pose", "rest_pose"),
        ),
        (".env.example", ("QUACKD_REGISTRY_DIR",)),
        ("docs/mcp.md", ("--flock",)),
        ("docs/memory.md", ("registered",)),
        ("SECURITY.md", ("robots.json",)),
    ):
        text = (REPO / path).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, f"{path} does not mention {needle!r}"


def test_every_command_is_named_in_the_readme_table_and_the_module_map() -> None:
    """`quackd memory` shipped in 0.6 and appeared in neither for a release. A command nobody
    can find is a command nobody has."""
    from quackd.cli import app

    # hidden commands are left out: `trace` is the 0.11 alias of `log` and is deliberately
    # absent from `--help`, so a README row for it would advertise the spelling being retired
    names = {
        c.name or (c.callback.__name__ if c.callback else "")
        for c in app.registered_commands
        if not c.hidden
    }
    names |= {g.name or "" for g in app.registered_groups}
    names = {n.replace("_", "-") for n in names if n}
    architecture = (REPO / "docs" / "architecture.md").read_text(encoding="utf-8")
    # the table row, not the prose: `quackd flock` is mentioned in three paragraphs and was
    # still missing from the one table somebody reads to find out that it exists
    rows = "".join(line for line in README.splitlines(keepends=True) if line.startswith("| `"))
    for name in sorted(names):
        assert f"| `quackd {name}" in rows, f"the README usage table has no row for {name}"
        assert name in architecture, f"docs/architecture.md never names {name}"


def _github_slug(heading: str) -> str:
    """The anchor GitHub mints for a heading, near enough to check links against.

    Backticks and inline links are stripped to their text, everything that is not a word
    character, a hyphen or a space goes, what is left is lowercased and its spaces become
    hyphens. That is GitHub's rule, and it is why `## The ones quackd names` is reached as
    `#the-ones-quackd-names`."""
    text = re.sub(r"`|<[^>]+>", "", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    """Every fragment a link may point at in this file: one per heading, plus any explicit
    `id=` or `name=`. A heading that repeats gets `-1`, `-2`, the way GitHub numbers them."""
    text = path.read_text(encoding="utf-8")
    seen: dict[str, int] = {}
    out: set[str] = set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*$", _prose(text), flags=re.M):
        slug = _github_slug(heading)
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        out.add(slug if n == 0 else f"{slug}-{n}")
    return out | set(re.findall(r'\b(?:id|name)="([^"]+)"', text))


def test_every_relative_link_and_anchor_in_the_markdown_resolves() -> None:
    """Every link between two files in this repository points at something that is there.

    Its predecessor checked one shape, `](adr/....md)`, and resolved it against `docs/`
    whatever file did the linking. So a page in a subdirectory writing `](../adr/....md)`
    was never checked at all, and one writing `](adr/....md)` passed while being broken on
    GitHub. That hole was invisible while every page sat directly under `docs/`, and stopped
    being invisible the day the decision LLM pages moved a level down.

    Anchors are checked too, because this is now a repository where one page links a heading
    on another, and a heading is renamed far more easily than a file.

    Links inside fenced blocks are not checked: those are examples of what a reader would
    type, and `_prose` takes them out for every other guard here as well.
    """
    anchors: dict[Path, set[str]] = {}
    broken: list[str] = []
    for md in sorted(REPO.glob("*.md")) + sorted((REPO / "docs").rglob("*.md")):
        for target in re.findall(r"\]\(([^)\s]+)\)", _prose(md.read_text(encoding="utf-8"))):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target):  # http, https, mailto, any scheme
                continue
            path, _, fragment = target.partition("#")
            dest = md if not path else (md.parent / path).resolve()
            if not dest.exists():
                broken.append(f"{md.relative_to(REPO)} -> {target} (no such file)")
            elif (
                fragment
                and dest.suffix == ".md"
                and fragment.lower() not in anchors.setdefault(dest, _anchors(dest))
            ):
                broken.append(f"{md.relative_to(REPO)} -> {target} (no such heading)")
    assert not broken, "links that go nowhere:\n" + "\n".join(broken)


# ── a page per decision LLM, and the row it has to agree with ───────────────────────────


@pytest.mark.parametrize("name", PRESET_NAMES)
def test_every_decision_llm_preset_has_a_page_that_agrees_with_its_row(name: str) -> None:
    """A decision LLM is a row of data, and its page is where a reader checks the row.

    The reason this is a test and not a convention: the hub's table once carried a `kev`
    install line that had lost `KEV_DTYPE=bf16` and the `uv run --extra serve` prefix the
    catalogue still had, and nothing in the suite could see it. A reader copying that cell
    ran bare `python` outside the synced environment. So every value a page quotes is read
    back off the page and compared with the row it came from.

    The three literals at the end are the honesty rule `docs/adapters.md` puts on an adapter
    page, applied to a server: say what you read and when, say what you are assuming, and
    keep the word never until somebody has actually run it.
    """
    spec = PRESETS[name]
    path = REPO / "docs" / "decision-llms" / f"{name}.md"
    assert path.exists(), f"every preset has a page: docs/decision-llms/{name}.md is missing"
    page = path.read_text(encoding="utf-8")
    for field in ("url", "model", "key_env", "install"):
        value = getattr(spec, field)
        if value is not None:
            assert value in page, f"{path.name} does not carry the row's {field}: {value!r}"
    if spec.extra:
        assert f"quackd[{spec.extra}]" in page, f"{path.name} does not name its extra"
    if spec.price is not None:
        assert f"{spec.price.input:g}" in page, f"{path.name} does not carry its published rate"
    assert f"--decision-llm {name}" in page, f"{path.name} does not show how to name it"
    assert "--decision-mode shadow" in page, f"{path.name} does not point at shadow mode"
    # As headings, and in that spelling: `"VERIFIED" in page` is satisfied by the word
    # UNVERIFIED, so the first half of this pair could not fail on any page that had the
    # second. The status line is checked literally for the same reason -- `never` on its own
    # is a word that turns up seven times in ordinary prose on one of these pages.
    for honesty in ("\n## VERIFIED", "\n## UNVERIFIED"):
        assert honesty in page, f"{path.name} has no {honesty.strip()} section"
    assert "**Nothing here has ever answered a real robot.**" in page, (
        f"{path.name} drops the status line before anybody has run it"
    )
    if name == "local":
        assert "--decision-url" in page and "/v1/systemone" in page


def test_the_decision_llms_hub_links_every_preset_page_from_its_table() -> None:
    """The table is the way in, so it carries the links, in the order `doctor` prints.

    It also carries each install line verbatim, which is the cell that drifted before: a
    table nobody reads against the code is a table that describes an older release.
    """
    hub = (REPO / "docs" / "decision-llms.md").read_text(encoding="utf-8")
    lines = hub.splitlines()
    rows: dict[str, int] = {}
    for name in PRESET_NAMES:
        row = next((i for i, line in enumerate(lines) if line.startswith(f"| `{name}` |")), None)
        assert row is not None, f"the hub's table has no row for {name}"
        rows[name] = row
        assert f"](decision-llms/{name}.md)" in lines[row], f"{name}'s row does not link its page"
        assert PRESETS[name].install in lines[row], f"{name}'s row does not quote its install line"
    assert list(rows) == sorted(rows, key=lambda n: rows[n]), "the table is not in doctor's order"
    linked = set(re.findall(r"\]\(decision-llms/([a-z_]+)\.md\)", hub))
    assert linked == set(PRESET_NAMES), f"the hub links {sorted(linked)}"


def test_no_living_document_claims_the_wrong_number_of_decision_llms() -> None:
    """Seven is `len(PRESET_NAMES)`, and the README says it in a status cell.

    The same shape as the cloud-provider guard above, and for the same reason: a count in
    prose is a fact about the code that nothing else would notice going stale."""
    right = len(PRESET_NAMES)
    wrong = [word for count, word in _NUMBER_WORDS.items() if count != right]
    for path in _living_docs():
        prose = _prose(path.read_text(encoding="utf-8")).lower()
        for word in wrong:
            assert f"{word} decision llms" not in prose, (
                f"{path.name} says {word} decision LLMs and the catalogue has {right}"
            )


#: TypeSafe's confidence page publishes exactly two numbers, 0.5 and 0.9, which are quackd's
#: brake and confirm-gated floors. The read floor at 0.60 and the motion floor at 0.85 are
#: quackd's own, set between those two, and nothing published sits there. The claim that all
#: four are theirs was written once and then copied onto nine pages and a README row, where it
#: outlived two rounds of editing, so it is a string now rather than a convention. A number
#: nobody published is a number nobody has calibrated either, and that is the whole reason
#: `--decision-mode shadow` exists.
_FLOORS_ARE_NOT_ALL_PUBLISHED = (
    "floors are jev's published numbers",
    "floors are jev's numbers",
    "these are jev's own published numbers",
    "floors on this page are jev's published numbers",
    "floors set on jev's published numbers",
    "floors are typesafe's published numbers",
    "every one of them a number typesafe publish",
    "typesafe's own universal floor",
    "confidence floors are typesafe's own published numbers",
)
#: An ADR body is what was believed on the day it was accepted, and both of these ADRs still
#: say it there. What has to be true today is the amendment note above the first heading,
#: which is the house's own correction mechanism, so that is the part read back -- minus
#: anything in quotation marks, because a note corrects a sentence by quoting it and a
#: checker that cannot tell a citation from a claim would forbid the fix along with the bug.


def test_no_living_document_credits_all_four_confidence_floors_to_typesafe() -> None:
    """Two of the four are quackd's own, and the docs have said otherwise twice.

    `https://docs.typesafe.ai/confidence` states 0.5 (genuinely unsure, route to a human) and
    0.9 (high stakes, proceed with confirmation), and says the right values are domain-specific.
    quackd's brake and confirm floors sit on those. Its read floor (0.60) and motion floor
    (0.85) sit between them and are nobody's published guidance, so a page that credits all
    four to a vendor is telling a reader those numbers carry an authority they do not have.

    The history files are exempt the way they always are, with one addition: ADR-0040 and
    ADR-0043 both state the old claim in their Consequences, and both now carry an amendment
    note correcting it. A body records what was believed and is left alone. The note is the
    live document, so the text above the ADR's first heading is what is read back, with
    quoted spans removed: both notes work by quoting the sentence they are overturning.
    """
    sources = [
        REPO / "quackd" / "agent" / "decision" / "stepper.py",
        REPO / "docs" / "adr" / "0040-a-discrete-stepper-in-front-of-the-model.md",
        REPO / "docs" / "adr" / "0043-decision-llms-are-a-wire-format-and-a-data-row.md",
    ]
    for path in _living_docs() + sources:
        text = path.read_text(encoding="utf-8")
        if "adr" in path.parts:
            # the metadata line and the amendment notes, which stop at the first section,
            # and not the sentences they quote in order to overturn them
            text = re.sub(r'"[^"]*"', "", text.split("\n## ", 1)[0])
        text = text.lower()
        for wrong in _FLOORS_ARE_NOT_ALL_PUBLISHED:
            assert wrong not in text, (
                f"{path.relative_to(REPO)} says {wrong!r}, but TypeSafe publish only 0.5 and "
                "0.9; the 0.60 read floor and the 0.85 motion floor are quackd's own"
            )
