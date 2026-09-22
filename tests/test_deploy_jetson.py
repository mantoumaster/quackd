"""`deploy/jetson/` says the same things the code and the page do, or one of them is wrong.

None of this can be run on a Jetson here, so what it checks is the half that is checkable from
Python: that the compose file describes a command rather than a service, that the model server
is bound to loopback and given the GPU, that the image installs extras that exist, and that the
JetPack table printed in the documentation is the one `quackd doctor` reads from. That last one
is the reason this file exists at all: two copies of a version table drift, and the copy people
paste into an issue is the one nobody re-reads.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

from quackd import doctor

REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "deploy" / "jetson"
DOCKERFILE = (DEPLOY / "Dockerfile").read_text(encoding="utf-8")
PAGE = (REPO / "docs" / "jetson.md").read_text(encoding="utf-8")
WORKFLOW = (REPO / ".github" / "workflows" / "jetson-image.yml").read_text(encoding="utf-8")

DOCKERFILE_CODE = "".join(
    line for line in DOCKERFILE.splitlines(keepends=True) if not line.lstrip().startswith("#")
)
"""The instructions without the comments.

The header explains at length why this image is not built FROM nvcr.io/nvidia/l4t-jetpack
and what CUDA would cost it. A check that grepped the whole file would read that
explanation as the thing it forbids."""


def _compose() -> dict[str, Any]:
    loaded = yaml.safe_load((DEPLOY / "compose.yml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _env(service: dict[str, Any]) -> dict[str, str]:
    """Compose takes a mapping or a list of `KEY=value`, and a test that knows only one of them
    would pass until somebody reformatted the file."""
    raw = service.get("environment", {})
    if isinstance(raw, list):
        return dict(item.split("=", 1) for item in raw)
    return {str(k): str(v) for k, v in raw.items()}


def test_the_quackd_service_is_a_command_and_not_a_service() -> None:
    """A restart policy on `quackd run` would re-run a robot task every time it finished and
    again at every boot, with nobody near the power switch. `bridge/open_duck`'s unit sets
    `Restart=no` for the same reason.

    `restart` is asserted against the string, not the value: a bare `no` is YAML's boolean
    false, which Compose reads as no policy by accident rather than on purpose."""
    quackd = _compose()["services"]["quackd"]
    assert quackd["restart"] == "no", "quote it, or YAML makes it a boolean"
    assert "run" in quackd["profiles"], "nothing in this file may start on a bare compose up"
    assert "depends_on" not in quackd, (
        "no depends_on: a board with Ollama installed natively must not get a second one "
        "started underneath it by a compose run"
    )
    assert quackd["stop_signal"] == "SIGINT", (
        "SIGTERM reaches no handler in quackd; SIGINT is the one the kill switch answers"
    )


def test_ollama_binds_loopback_and_owns_the_gpu() -> None:
    ollama = _compose()["services"]["ollama"]
    assert _env(ollama)["OLLAMA_HOST"] == "127.0.0.1:11434", (
        "the image's own default is 0.0.0.0, which under host networking is every interface"
    )
    assert ollama["runtime"] == "nvidia", (
        "the deploy.resources.devices form does not reliably reach the Tegra integrated GPU"
    )
    assert ollama["restart"] == "unless-stopped", "this one is a service, and comes back at boot"
    assert "ollama" in ollama["profiles"]


def test_only_the_model_server_asks_for_a_gpu() -> None:
    """The whole argument of the page in one assertion: quackd is the CPU half."""
    quackd = _compose()["services"]["quackd"]
    assert "runtime" not in quackd and "devices" not in quackd
    assert "deploy" not in quackd


def test_both_services_share_the_board_s_network() -> None:
    """`--llm ollama` means http://localhost:11434/v1, and a robot daemon on this same board is
    127.0.0.1 as well. Neither is true from inside a bridge network."""
    services = _compose()["services"]
    assert {s["network_mode"] for s in services.values()} == {"host"}


def test_the_image_extras_all_exist() -> None:
    # an extra name, not everything up to the next space: the ARG default is quoted, and
    # the closing quote rode along into the name the first time this was written
    extras = set(re.findall(r"--extra ([A-Za-z0-9._-]+)", DOCKERFILE_CODE))
    declared = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    unknown = extras - set(declared)
    assert not unknown, f"the Dockerfile installs extras quackd does not publish: {unknown}"
    assert {"openai", "microduck"} <= extras, (
        "openai is the client every local preset speaks through, and a bare quackd carries "
        "no robot at all"
    )
    assert "yolo" not in extras, (
        "ultralytics is torch and it is AGPL, and nothing in this image would give it a GPU"
    )


def test_the_dockerfile_pins_every_image_and_carries_no_cuda() -> None:
    images = re.findall(r"^FROM (\S+)", DOCKERFILE_CODE, flags=re.M)
    images += re.findall(r"COPY --from=(\S+/\S+) ", DOCKERFILE_CODE)
    # both stages are FROM ${PYTHON_IMAGE}, so a check that skipped unresolved names
    # checked everything except the base image this is all built on
    args = dict(re.findall(r"^ARG (\w+)=(\S+)", DOCKERFILE_CODE, flags=re.M))
    concrete = [
        re.sub(r"\$\{(\w+)\}", lambda m: args.get(m.group(1), m.group(0)), image)
        for image in images
    ]
    assert len(concrete) >= 3, concrete
    for image in concrete:
        assert "${" not in image, f"{image} names an ARG with no default to resolve"
        assert ":" in image and not image.endswith(":latest"), f"{image} is not pinned"
    for banned in ("nvcr.io", "l4t-", "cuda"):
        assert banned not in DOCKERFILE_CODE.lower(), (
            f"{banned!r} in the image: quackd needs no CUDA, and the GPU belongs to the "
            "model server beside it"
        )


def test_the_dockerignore_keeps_the_env_file_out() -> None:
    """The build copies the repository root, and a copied secret does not stop existing when a
    later layer deletes it: it is in the build cache, and CI uploads that cache."""
    ignored = [
        line.strip()
        for line in (REPO / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    # patterns, not the prose about them: splitting the whole file on whitespace let the
    # comment that explains these rules satisfy a test for the rules themselves
    for needle in ("**/.env", "**/.env.*", ".git", "runs/", ".venv"):
        assert needle in ignored, f".dockerignore does not exclude {needle}"
    assert ".env" not in ignored, (
        "a bare .env is anchored at the context root and misses deploy/jetson/.env, "
        "which is where the compose file tells a Jetson user to put a key"
    )


def test_the_page_and_the_compose_file_name_the_same_model() -> None:
    """A page that tells you to pull one model and a file that runs another is a page that
    fails on the reader's board and not on ours."""
    default = _env(_compose()["services"]["quackd"])["QUACKD_LLM"]
    model = re.fullmatch(r"\$\{QUACKD_LLM:-ollama:(?P<model>[\w.:-]+)\}", default)
    assert model, f"QUACKD_LLM is not an ollama spec with a default: {default!r}"
    tag = model.group("model")
    assert f"ollama pull {tag}" in PAGE, f"docs/jetson.md never says to pull {tag}"
    assert f"ollama pull {tag}" in (DEPLOY / "README.md").read_text(encoding="utf-8")


def test_the_jetpack_table_matches_the_one_doctor_reads() -> None:
    """Two copies of a version table drift, and the copy people paste into an issue is the one
    nobody re-reads. The page's table is the documentation; `_JETPACK_FOR_L4T` is what the
    command prints. They are the same fact and this is the only thing holding them together."""
    rows = dict(re.findall(r"^\| `r(\d[\w.]*)` \| ([\w.]+) \|", PAGE, flags=re.M))
    assert rows, "no L4T table found in docs/jetson.md, or its shape changed"
    assert rows == doctor._JETPACK_FOR_L4T, (
        "docs/jetson.md and quackd/doctor.py disagree about which JetPack an L4T release is:\n"
        f"  page:   {sorted(rows.items())}\n"
        f"  doctor: {sorted(doctor._JETPACK_FOR_L4T.items())}"
    )


def test_the_workflow_builds_and_never_publishes() -> None:
    """An image nobody has run on a Jetson is not something to put a pull command beside. The
    job proves a checkout builds; it does not hand anyone an artifact."""
    workflow = yaml.safe_load(WORKFLOW)
    job = workflow["jobs"]["image"]
    assert job["runs-on"] == "ubuntu-24.04-arm", "native arm64, so the smoke steps are not QEMU"
    assert workflow["permissions"] == {"contents": "read"}, "no packages: write, nothing is pushed"
    build = next(s for s in job["steps"] if str(s.get("uses", "")).startswith("docker/build-push"))
    assert build["with"]["push"] is False
    assert build["with"]["platforms"] == "linux/arm64"
    assert "ghcr.io" not in WORKFLOW and "docker/login-action" not in WORKFLOW


def test_the_workflow_reruns_on_the_files_it_actually_builds_from() -> None:
    """And deliberately not on the core, which its own header explains.

    A job whose smoke steps assert against `quackd/` but whose filters exclude it can rot
    quietly. That is a considered trade rather than an oversight: `ci` runs the same
    assertions natively on every push. This test is what keeps the two halves in step."""
    workflow = yaml.safe_load(WORKFLOW)
    # "on" is YAML 1.1's boolean true, so the key may have been parsed either way
    triggers = workflow.get("on") or workflow.get(True)
    assert triggers, "the workflow has no trigger block, or its shape changed"
    for event in ("pull_request", "push"):
        paths = triggers[event]["paths"]
        for needed in ("deploy/jetson/**", ".dockerignore", "uv.lock", "pyproject.toml"):
            assert needed in paths, f"{event} does not rerun on {needed}"
        assert "quackd/**" not in paths, (
            "if this is added, drop the paragraph in the workflow header that explains why "
            "it is absent"
        )


def test_the_page_says_no_jetson_has_run_this() -> None:
    """The claim this whole change turns on, and the one a well meaning edit would soften. It
    is spelled this way rather than any of the retired phrasings `tests/test_docs.py` bans,
    because an SO-101 arm has run quackd and a Jetson has not."""
    # the whole sentence, because the negation is in the first word: the substring
    # "run on a Jetson by this project" is just as true of a page claiming the opposite
    for path, text, sentence in (
        ("docs/jetson.md", PAGE, "Nothing here has been run on a Jetson by this project"),
        ("docs/jetson.md", PAGE, "Nothing on this page has been run on a Jetson by this project"),
        (
            "deploy/jetson/README.md",
            (DEPLOY / "README.md").read_text(encoding="utf-8"),
            "Nothing here has been run on a Jetson by this project",
        ),
    ):
        assert sentence in text, f"{path} no longer says: {sentence}"
