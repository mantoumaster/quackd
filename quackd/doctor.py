"""`quackd doctor`: what can run here, and what this machine is assuming about the robot.

It exists because "it doesn't work" almost always means a missing extra, a missing key, or
an upstream assumption, and all three should be visible in one screen before anyone opens
an issue.

It is two halves on purpose. `collect` asks the questions — which modules import, which keys
are set, which local servers answer, what a real robot says about itself — and answers in
dataclasses with no styling anywhere in them. `render` decides what that looks like, and
`to_dict` is the same report with no renderer at all. That split is what `--json` is made
of, and it was not possible before: every cell in here used to *be* a markup string, so
there was nothing underneath to serialise.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.rule import Rule
from rich.text import Text

from quackd import __version__, ui
from quackd.adapters.base import AdapterError
from quackd.adapters.factory import describe, list_adapters, parse_robot_spec
from quackd.agent.providers.base import ProviderError
from quackd.agent.providers.factory import (
    EXTRA_FOR,
    KEY_ENV,
    LOCAL_NAMES,
    PROVIDER_NAMES,
    SDK_FOR,
    default_model,
    resolve_model,
)
from quackd.agent.providers.local import PRESETS
from quackd.duckfile.parser import list_bundled_ducks
from quackd.transport import upstream_api as up
from quackd.transport.factory import TRANSPORT_STATUS

# The optional extras table, which is about packages rather than providers: the providers table
# builds its own rows from SDK_FOR and EXTRA_FOR. One wheel now serves nine vendors, so naming
# them all here would be a list to keep in step for no gain; `quackd list-models` and the
# providers table above already say which vendor wants which install.
EXTRAS = {
    "anthropic": ("anthropic", "quackd[anthropic]"),
    "openai": ("openai", "quackd[openai] and every OpenAI-compatible vendor"),
    "gemini": ("google.genai", "quackd[gemini]"),
    "yolo": ("ultralytics", "quackd[yolo]"),
    "live": ("pygame", "quackd[live]"),
    "mujoco": ("mujoco", "quackd[mujoco]"),
    "lan (zeroconf)": ("zeroconf", "quackd[lan]"),
    "lan (mqtt)": ("paho.mqtt.client", "quackd[lan]"),
    "lerobot": ("lerobot", "quackd[lerobot]"),
    "lerobot (feetech bus)": ("scservo_sdk", "quackd[lerobot]"),
    "rosbridge": ("roslibpy", "quackd[rosbridge]"),
    "microduck camera (webrtc)": ("aiortc", "quackd[microduck-camera]"),
    "xlerobot": ("zmq", "quackd[xlerobot]"),
    "alohamini": ("zmq", "quackd[alohamini]"),
}
# Robot SDKs are looked up by distribution metadata only: importing lerobot pulls torch
# into a diagnostics command, which is exactly what doctor is not. The Feetech SDK is the
# half of `quackd[lerobot]` that opens the serial port, and a lerobot installed without its
# `[feetech]` extra imports cleanly and then cannot reach an arm, so doctor asks for it by name.
_METADATA_ONLY = {"lerobot": "lerobot", "scservo_sdk": "feetech-servo-sdk"}

CORE_MODULES = (
    ("pydantic", "pydantic"),
    ("mcp", "mcp"),
    ("opencv", "cv2"),
    ("numpy", "numpy"),
    ("Pillow", "PIL"),
)

Progress = Any
"""`progress(message)`, or None: what the spinner says while a slow question is asked."""

_ADAPTER_HINT = "quackd list-adapters shows the seven that ship and their backends"

FLOCK_NOTE = (
    "flock mode (--flock, flock.roles): sim2d only, in-process bus by default. The MQTT bus "
    "(quackd[lan]) is library-only (docs/lan.md)."
)


def _installed(module: str) -> str | None:
    if module in _METADATA_ONLY:
        try:
            return md.version(_METADATA_ONLY[module])
        except md.PackageNotFoundError:
            return None
    try:
        importlib.import_module(module)
    except Exception:
        return None
    dist = {"google.genai": "google-genai", "paho.mqtt.client": "paho-mqtt"}.get(module, module)
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        pass
    # An import name is not a distribution name: `cv2` ships as opencv-python-headless here
    # and as opencv-python elsewhere, `PIL` as pillow. Ask the installer which one it was,
    # rather than printing "?" next to a package that is plainly installed and working.
    for candidate in md.packages_distributions().get(module.split(".")[0], []):
        try:
            return md.version(candidate)
        except md.PackageNotFoundError:
            continue
    return "?"


def _mask(value: str) -> str:
    return value[:4] + "…" + value[-2:] if len(value) > 8 else "set"


# ── what the report is made of ──────────────────────────────────────────────────────────


@dataclass
class Check:
    """One yes-or-no fact with its evidence: a module and its version, an extra and how to
    install it."""

    name: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class ProviderRow:
    name: str
    extra: str
    version: str | None
    key: str
    """Masked, or empty when there is none."""
    key_env: str
    key_optional: bool
    """A local server does not need one; a cloud provider cannot run without it."""
    model: str
    pinned: bool = False
    """QUACKD_MODEL chose this one, rather than the vendor's default."""
    refused_model: str = ""
    """A QUACKD_MODEL this vendor does not list, which is why `model` is its default and not
    what the environment asked for."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "extra": self.extra,
            "version": self.version,
            "key": self.key or None,
            "key_env": self.key_env,
            "key_optional": self.key_optional,
            "model": self.model,
            "pinned": self.pinned,
            "refused_model": self.refused_model or None,
        }


@dataclass
class ServerRow:
    preset: str
    url: str
    state: str
    """up · http · down · unset"""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"preset": self.preset, "url": self.url, "state": self.state, "detail": self.detail}


@dataclass
class VerbRow:
    name: str
    core: bool
    safety: str
    preconditions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "core": self.core,
            "safety": self.safety,
            "preconditions": self.preconditions,
        }


@dataclass
class ProbeRow:
    what: str
    value: str
    state: str = "plain"
    """plain · ok · warn · fail"""

    def to_dict(self) -> dict[str, Any]:
        return {"what": self.what, "value": self.value, "state": self.state}


@dataclass
class ProbeReport:
    """What a real robot said about itself.

    The only part of doctor that leaves this machine, and the only way to see the difference
    between quackd's description of a fully built robot and the one somebody assembled,
    before a run finds it."""

    address: str
    ok: bool
    rows: list[ProbeRow] = field(default_factory=list)
    advisories: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "ok": self.ok,
            "rows": [r.to_dict() for r in self.rows],
            "advisories": self.advisories,
            "error": self.error,
        }


@dataclass
class RobotReport:
    spec: str
    summary: str = ""
    verbs: list[VerbRow] = field(default_factory=list)
    probe: ProbeReport | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "summary": self.summary,
            "verbs": [v.to_dict() for v in self.verbs],
            "probe": self.probe.to_dict() if self.probe else None,
            "error": self.error,
        }


@dataclass
class TransportRow:
    name: str
    status: str
    note: str = ""
    found: bool = False
    """Something this transport needs is here: a robotd socket on the machine, say."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "note": self.note,
            "found": self.found,
        }


@dataclass
class Assumption:
    upstream: str
    what: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {"upstream": self.upstream, "what": self.what, "note": self.note}


@dataclass
class PinRow:
    """Where an upstream was read, when, and how much of it anybody has actually run."""

    upstream: str
    pin: str
    extra_pin: str = ""
    """A second commit where an upstream has one: microduck_rl pins its policies apart from
    its model, and they move independently."""
    read_on: str = ""
    verified: int = 0
    unverified: int = 0
    never_run: str = ""
    doc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "upstream": self.upstream,
            "pin": self.pin,
            "extra_pin": self.extra_pin,
            "read_on": self.read_on,
            "verified": self.verified,
            "unverified": self.unverified,
            "never_run": self.never_run,
            "doc": self.doc,
        }


@dataclass
class DoctorReport:
    version: str
    python: str
    platform: str
    api_version: str = ""
    """The duck-ipc-proto version quackd speaks. The one number here that is about the wire
    rather than about this machine, and the first thing to check against a robot."""
    core: list[Check] = field(default_factory=list)
    bundled_ducks: int = 0
    providers: list[ProviderRow] = field(default_factory=list)
    servers: list[ServerRow] = field(default_factory=list)
    adapters: list[dict[str, Any]] = field(default_factory=list)
    transports: list[TransportRow] = field(default_factory=list)
    extras: list[Check] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    pins: list[PinRow] = field(default_factory=list)
    robot: RobotReport | None = None

    @property
    def ok(self) -> bool:
        """Whether this machine is in a state to run anything: the core imports, and the
        robot the command was asked about answered. A missing extra is a choice, not a
        fault, and never fails the check."""
        if any(not c.ok for c in self.core):
            return False
        if self.robot is not None and self.robot.error is not None:
            return False
        return not (self.robot and self.robot.probe and not self.robot.probe.ok)

    @property
    def missing_core(self) -> list[str]:
        return [c.name for c in self.core if not c.ok]

    @property
    def cloud_keys(self) -> list[str]:
        """Providers that would run here if they had a key."""
        return [p.name for p in self.providers if not p.key_optional and not p.key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "version": self.version,
            "python": self.python,
            "platform": self.platform,
            "api_version": self.api_version,
            "core": [c.to_dict() for c in self.core],
            "bundled_ducks": self.bundled_ducks,
            "providers": [p.to_dict() for p in self.providers],
            "servers": [s.to_dict() for s in self.servers],
            "adapters": self.adapters,
            "transports": [t.to_dict() for t in self.transports],
            "extras": [e.to_dict() for e in self.extras],
            "assumptions": [a.to_dict() for a in self.assumptions],
            "pins": [p.to_dict() for p in self.pins],
            "robot": self.robot.to_dict() if self.robot else None,
        }


# ── asking ──────────────────────────────────────────────────────────────────────────────


def _probe_models(base_url: str, timeout_s: float = 1.5) -> tuple[str, str]:
    """Reachability of an OpenAI-compatible server, as (state, detail)."""
    import json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=timeout_s) as r:
            payload = json.loads(r.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        return "http", f"HTTP {e.code}"
    except Exception:
        return "down", "not running"
    ids = [str(m.get("id", "")) for m in payload.get("data", []) if isinstance(m, dict)]
    shown = ", ".join(i for i in ids[:3] if i)
    more = f" (+{len(ids) - 3})" if len(ids) > 3 else ""
    return "up", f"{shown}{more}" if ids else "no models loaded"


def probe(
    spec: str,
    static: Any,
    address: str,
    camera_url: str | None,
    token: str | None,
) -> ProbeReport:
    """Connect, and report what the robot itself said.

    Everything else in this file is offline and reads the *static* manifest, which describes
    a fully built robot. A real one is whatever its owner assembled, so this is the only way
    to see the difference before a run does."""
    import asyncio

    from quackd.adapters.factory import make_adapter
    from quackd.transport.base import TransportError

    async def go() -> tuple[Any, Any, dict[str, Any] | None, dict[str, Any]]:
        adapter = make_adapter(
            parse_robot_spec(spec), address=address, camera_url=camera_url, token=token
        )
        live = await adapter.connect()
        transport = getattr(adapter, "transport", None)
        # What the robot says about its own guarantees, rather than what quackd's static
        # description claims on its behalf. This is the checklist's go/no-go gate, so a
        # deadman window or an absent token has to be visible here.
        told: dict[str, Any] = dict(getattr(transport, "safety", None) or {})
        for key in ("auth_warning", "runtime_warning"):
            if warning := getattr(transport, key, None):
                told[key] = warning
        if commit := getattr(transport, "runtime_commit", None):
            told["runtime_commit"] = commit
        try:
            health = await adapter.health()
            # A camera URL that nothing checks is a camera URL that fails mid-run. doctor used
            # to accept --camera-url, hand it to the transport and never ask for a frame, so a
            # typo'd or unreachable snapshot server passed here and failed at the first observe.
            camera: dict[str, Any] | None = None
            cam_probe = getattr(transport, "camera_health", None)
            # Only report on a camera the adapter actually reads. `camera_url` is accepted and
            # ignored by rosbridge, and gating its verdict on a frame from an unrelated path
            # fails a healthy robot.
            if camera_url and callable(cam_probe):
                # Frames arrive on a timer, so ask for one and give the capture loop a moment
                # rather than reading memory that cannot have been filled yet.
                frame = await adapter.get_frame()
                for _ in range(50):
                    if frame is not None:
                        break
                    await asyncio.sleep(0.1)
                    frame = await adapter.get_frame()
                camera = dict(cam_probe())
                camera["frame"] = f"{frame.width}x{frame.height}" if frame is not None else None
            return live, health, camera, told
        finally:
            await adapter.disconnect()

    try:
        live, health, camera, told = asyncio.run(go())
    except (TransportError, OSError) as e:
        return ProbeReport(address=address, ok=False, error=f"{spec} at {address}: {e}")

    report = ProbeReport(address=address, ok=True)
    add = report.rows.append
    add(ProbeRow("connected", "yes", "ok"))
    why = "ok" if health.ok else str(health.reason or "not ok, and it did not say why")
    add(ProbeRow("health", why, "ok" if health.ok else "fail"))
    for key, value in (health.extras or {}).items():
        add(ProbeRow(f"  {key}", "" if value is None else str(value)))
    gained = sorted(set(live.verb_names()) - set(static.verb_names()))
    lost = sorted(set(static.verb_names()) - set(live.verb_names()))
    add(ProbeRow("verbs", f"{len(live.verb_names())} of {len(static.verb_names())} described"))
    if lost:
        add(ProbeRow("  not on this robot", ", ".join(lost), "warn"))
    if gained:
        add(ProbeRow("  beyond the description", ", ".join(gained), "ok"))
    for key, value in (live.extras.get("expression_features") or {}).items():
        add(ProbeRow(f"  {key}", "yes" if value else "no", "ok" if value else "plain"))
    if told:
        # The deadman window is a free parameter and whether there is a token at all is the
        # difference between the documented setup and an open port, so both belong in front
        # of the operator at the checklist's go/no-go gate.
        add(ProbeRow("safety", "as this bridge reported it"))
        for key in (
            "deadman_ms",
            "auth",
            "fall_detection",
            "getup_policy",
            "estop",
            "runtime_commit",
        ):
            if key in told:
                reported = told[key]
                worrying = (key == "auth" and reported == "none") or (
                    key in ("fall_detection", "getup_policy") and reported is False
                )
                add(ProbeRow(f"  {key}", str(reported), "warn" if worrying else "plain"))
    camera_ok = True
    if camera is not None:
        frame = camera.get("frame")
        camera_ok = frame is not None
        shown = str(frame) if camera_ok else "no frame"
        add(ProbeRow("camera", shown, "ok" if camera_ok else "fail"))
        add(ProbeRow("  url", str(camera.get("url") or camera_url)))
        if camera.get("error"):
            add(ProbeRow("  error", str(camera["error"]), "fail"))
    if lost:
        report.advisories.append(
            f"a .duck that requires {lost[0]} will be refused on this robot, and one that "
            "merely allows it runs without it"
        )
    if not camera_ok:
        # naming verbs this body does not have sends the reader looking for them at the
        # moment they are trying to work out why their webcam gave nothing
        from quackd.verbs.core import REQUIREMENTS

        blind = [
            name
            for name in live.verb_names()
            if (req := REQUIREMENTS.get(name)) is not None and req.camera
        ]
        report.advisories.append(
            "--camera-url was given but no frame came back, so "
            + (", ".join(blind) if blind else "nothing that needs a camera")
            + " cannot see anything on this run"
        )
    for key in ("auth_warning", "runtime_warning"):
        if warning := told.get(key):
            report.advisories.append(str(warning))
    if told.get("fall_detection") is False:
        report.advisories.append(
            "nothing on this robot detects a fall, so posture never becomes 'fallen' and no "
            "verb refuses because it is down. You are the fall detector: keep it on a stand "
            "and watch it."
        )
    report.ok = bool(health.ok) and camera_ok
    return report


def _upstreams() -> list[tuple[str, Any, str, str]]:
    """(name, module, doc, what nobody has run it against). Imported in here rather than at
    module scope because doctor must not pull in an SDK to answer a question about it."""
    from quackd.adapters.alohamini import upstream_api as alohamini_api
    from quackd.adapters.lerobot import upstream_api as lerobot_api
    from quackd.adapters.open_duck import upstream_api as open_duck_api
    from quackd.adapters.rosbridge import upstream_api as rosbridge_api
    from quackd.adapters.toddlerbot import upstream_api as toddlerbot_api
    from quackd.adapters.xlerobot import upstream_api as xlerobot_api
    from quackd.sim3d import upstream_api as rl

    return [
        ("microduck", up, "docs/adapter-status.md", "a robotd (the jsonrpc backend)"),
        ("lerobot", lerobot_api, "docs/adapters/lerobot.md", "an arm (the real backend)"),
        ("rosbridge", rosbridge_api, "docs/adapters/rosbridge.md", "a bridge (the ws backend)"),
        ("open_duck", open_duck_api, "docs/adapters/open_duck.md", "a duck (the bridge backend)"),
        ("xlerobot", xlerobot_api, "docs/adapters/xlerobot.md", "a cart (the zmq backend)"),
        ("alohamini", alohamini_api, "docs/adapters/alohamini.md", "a robot (the zmq backend)"),
        ("toddlerbot", toddlerbot_api, "docs/adapters/toddlerbot.md", "a humanoid (the bridge)"),
        (
            "microduck_rl",
            rl,
            "docs/adr/0030-mujoco-physics-backend.md",
            "a robot: the model and the policies are fetched at run time and never shipped",
        ),
    ]


def collect(
    robot: str | None = None,
    *,
    address: str | None = None,
    camera_url: str | None = None,
    token: str | None = None,
    progress: Progress = None,
) -> DoctorReport:
    """Every question doctor asks, answered as data.

    Nothing in here decides what anything looks like, which is what lets `--json` exist and
    what keeps `render` honest about where its numbers came from."""

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    report = DoctorReport(
        version=__version__,
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.release()}",
        api_version=str(up.API_VERSION.name),
    )

    say("checking the core packages")
    for name, module in CORE_MODULES:
        version = _installed(module)
        report.core.append(Check(name, version is not None, version or "missing"))
    report.bundled_ducks = len(list_bundled_ducks())

    say("checking the providers")
    for name in PROVIDER_NAMES:
        if name == "fake":
            report.providers.append(
                ProviderRow("fake", "built-in", "built-in", "", "", True, "scripted")
            )
            continue
        key = os.environ.get(KEY_ENV[name], "")
        # What this provider would actually be given, not what the table used to guess. A
        # QUACKD_MODEL meant for one vendor is refused by the others, and doctor is where a
        # reader should find that out rather than three commands later.
        refused = ""
        try:
            model = resolve_model(name, default_model(name), source="QUACKD_MODEL") or (
                "auto (first served)"
            )
        except ProviderError:
            model = default_model(name) or "auto (first served)"
            refused = str(os.environ.get("QUACKD_MODEL", ""))
        report.providers.append(
            ProviderRow(
                name=name,
                extra=f"quackd[{EXTRA_FOR[name]}]",
                version=_installed(SDK_FOR[name]),
                key=_mask(key) if key else "",
                key_env=KEY_ENV[name],
                key_optional=name in LOCAL_NAMES,
                model=model,
                pinned=bool(os.environ.get("QUACKD_MODEL")) and not refused,
                refused_model=refused,
            )
        )

    custom = os.environ.get("QUACKD_BASE_URL")
    for preset, url in {**PRESETS, **({"local": custom} if custom else {})}.items():
        if not url:
            report.servers.append(
                ServerRow(preset, "", "unset", "set QUACKD_BASE_URL or --base-url")
            )
            continue
        say(f"probing {preset} at {url}")
        state, detail = _probe_models(url)
        report.servers.append(ServerRow(preset, url, state, detail))

    report.adapters = list_adapters()

    if robot is not None:
        say(f"describing {robot}")
        try:
            manifest = describe(parse_robot_spec(robot))
        except AdapterError as e:
            report.robot = RobotReport(spec=robot, error=str(e))
        else:
            report.robot = RobotReport(
                spec=robot,
                summary=manifest.summary(),
                verbs=[
                    VerbRow(
                        name=v.name,
                        core=bool(v.core),
                        safety=v.safety_class,
                        preconditions=list(manifest.preconditions.get(v.name, [])),
                    )
                    for v in manifest.verbs
                ],
            )
            if address:
                say(f"connecting to {robot} at {address}")
                report.robot.probe = probe(robot, manifest, address, camera_url, token)

    for name, status in TRANSPORT_STATUS.items():
        note, found = "", False
        if name == "jsonrpc":
            root = os.environ.get(up.RUNTIME_DIR_ENV.name, "/run")
            sock = Path(root) / "robotd.sock"
            if sys.platform == "win32":
                note = (
                    "Windows: use --address tcp://host:port via "
                    "`ssh -L 9870:/run/robotd.sock robot`"
                )
            elif sock.exists():
                note = f"{sock} present"
                found = True
            else:
                note = f"{sock} not found (not on a robot?)"
        if name == "websocket":
            note = up.WEBSOCKET_GATEWAY.note
        report.transports.append(TransportRow(name, status, note, found))

    say("checking the optional extras")
    for label, (module, extra) in EXTRAS.items():
        version = _installed(module)
        report.extras.append(
            Check(label, version is not None, version or f"not installed ({extra})")
        )

    for name, api, doc, never in _upstreams():
        unverified = api.refs_by_status("UNVERIFIED")
        for ref in unverified:
            report.assumptions.append(Assumption(name, ref.name, ref.note))
        report.pins.append(
            PinRow(
                upstream=name,
                pin=api.PIN[:7],
                extra_pin=getattr(api, "POLICIES_PIN", "")[:7],
                read_on=api.READ_ON,
                verified=len(api.refs_by_status("VERIFIED")),
                unverified=len(unverified),
                never_run=never,
                doc=doc,
            )
        )
    return report


# ── showing ─────────────────────────────────────────────────────────────────────────────

_STATE_STYLE = {"ok": "ok", "up": "ok", "warn": "warn", "http": "warn", "fail": "fail"}


def _section(console: Console, title: str) -> None:
    """A rule rather than a table title. There are ten sections here and they used to arrive
    as ten stacked tables with nothing between them, which read as one long table."""
    console.print()
    console.print(Rule(Text(title, style=ui.STYLES["key"]), align="left", style=ui.STYLES["rule"]))


def _checks(checks: list[Check], *, missing_is_fine: bool = False) -> Any:
    """A yes-or-no list. A missing extra is a choice; a missing core package is a fault."""

    def build(g: ui.Glyphs) -> Any:
        rows: list[tuple[str, Any]] = []
        for check in checks:
            if check.ok:
                mark, style = g.ok, ui.STYLES["ok"]
            elif missing_is_fine:
                mark, style = g.note, ui.STYLES["muted"]
            else:
                mark, style = g.fail, ui.STYLES["fail"]
            rows.append((f"{mark} {check.name}", Text(check.detail, style=style)))
        return ui.kv_grid(rows, key_style="")

    return ui.Deferred(build)


def _providers_table(report: DoctorReport) -> Any:
    table = ui.table()
    table.add_column("provider", style=ui.STYLES["key"], no_wrap=True)
    table.add_column("extra")
    table.add_column("key")
    # folded, not elided: a model id with an ellipsis through it cannot be pasted into --model
    table.add_column("default model", overflow="fold")
    for row in report.providers:
        if row.name == "fake":
            table.add_row(
                Text("fake"), Text("built-in", style=ui.STYLES["ok"]), Text("-"), Text(row.model)
            )
            continue
        extra = (
            Text(str(row.version), style=ui.STYLES["ok"])
            if row.version
            else Text.assemble(("missing ", ui.STYLES["warn"]), (f"({row.extra})", ""))
        )
        key: Any
        if row.key:
            key = ui.plain(row.key, style=ui.STYLES["ok"])
        elif row.key_optional:
            key = Text("optional", style=ui.STYLES["muted"])
        else:
            key = Text(f"{row.key_env} unset", style=ui.STYLES["warn"])
        if row.refused_model:
            model = Text.assemble(
                (f"QUACKD_MODEL={row.refused_model}", ui.STYLES["warn"]),
                (f", which {row.name} does not list", ui.STYLES["warn"]),
            )
        else:
            model = Text(row.model, style=ui.STYLES["ok"] if row.pinned else "")
        table.add_row(Text(row.name), extra, key, model)
    return table


def _servers_table(report: DoctorReport) -> Any:
    table = ui.table()
    table.add_column("preset", style=ui.STYLES["key"], no_wrap=True)
    table.add_column("base url")
    table.add_column("status")
    for row in report.servers:
        if row.state == "unset":
            table.add_row(Text(row.preset), Text(row.detail, style=ui.STYLES["muted"]), Text(""))
            continue
        if row.state == "up":
            # the detail is the model ids, which are the answer; the word is the good news
            status = Text("up", style=ui.STYLES["ok"])
            status.append(f" {row.detail}", style=ui.STYLES["muted"])
        else:
            # the detail already says it ("not running", "HTTP 500"), so the word would be
            # the same thing twice
            status = Text(row.detail, style=ui.STYLES[_STATE_STYLE.get(row.state, "muted")])
        table.add_row(Text(row.preset), Text(row.url), status)
    return table


def _verbs_table(robot: RobotReport) -> Any:
    table = ui.table(f"{robot.spec}: {robot.summary}")
    table.add_column("verb", style=ui.STYLES["key"], no_wrap=True)
    table.add_column("core")
    table.add_column("safety")
    table.add_column("preconditions")
    for verb in robot.verbs:
        table.add_row(
            Text(verb.name),
            Text("core" if verb.core else ""),
            Text(
                verb.safety, style=ui.STYLES["ok"] if verb.safety == "safe" else ui.STYLES["warn"]
            ),
            Text(", ".join(verb.preconditions), style=ui.STYLES["muted"]),
        )
    return table


def _probe_table(probe_report: ProbeReport) -> Any:
    table = ui.table(f"at {probe_report.address}: what the robot itself reported")
    table.add_column("what", style=ui.STYLES["key"], no_wrap=True)
    table.add_column("value")
    for row in probe_report.rows:
        style = ui.STYLES.get(_STATE_STYLE.get(row.state, ""), "")
        table.add_row(Text(row.what), Text(row.value, style=style))
    return table


def _assumptions_table(report: DoctorReport) -> Any:
    """One table for eight upstreams, sectioned. It was eight tables and eight dim footers,
    and a screen of them buried the one line anybody had come to read."""
    table = ui.table()
    table.add_column("upstream", style=ui.STYLES["key"], no_wrap=True)
    table.add_column("what")
    table.add_column("note", ratio=2)
    last = ""
    for item in report.assumptions:
        if last and item.upstream != last:
            table.add_section()
        table.add_row(
            Text(item.upstream if item.upstream != last else ""),
            Text(item.what),
            ui.plain(item.note, style=ui.STYLES["muted"]),
        )
        last = item.upstream
    return table


def _read_more(report: DoctorReport) -> Any:
    """Where to read about each upstream.

    A line rather than a column: eight table titles used to carry these paths, and a title
    is full width. Folded into a sixth column of the pins table, a path loses its tail on
    any terminal under 120, and a path you cannot copy is not a path."""
    parts = ["read more:", *(f"{pin.upstream} {pin.doc}" for pin in report.pins)]
    return ui.Deferred(lambda g: ui.joined(parts, g))


def _pins_table(report: DoctorReport) -> Any:
    table = ui.table("where each upstream was read, and what nobody has run it against")
    table.add_column("upstream", style=ui.STYLES["key"], no_wrap=True)
    table.add_column("pinned at", no_wrap=True)
    table.add_column("read on", no_wrap=True)
    table.add_column("refs", no_wrap=True)
    table.add_column("never run against")
    for pin in report.pins:
        refs = Text.assemble(
            (f"{pin.verified} verified", ui.STYLES["ok"]),
            ", ",
            (f"{pin.unverified} not", ui.STYLES["warn"] if pin.unverified else ui.STYLES["muted"]),
        )
        pinned = Text(pin.pin)
        if pin.extra_pin:
            pinned.append(f", policies {pin.extra_pin}", style=ui.STYLES["muted"])
        table.add_row(
            Text(pin.upstream),
            pinned,
            Text(pin.read_on, style=ui.STYLES["muted"]),
            refs,
            ui.plain(pin.never_run, style=ui.STYLES["muted"]),
        )
    return table


def _transports_table(report: DoctorReport) -> Any:
    table = ui.table()
    table.add_column("name", style=ui.STYLES["key"], no_wrap=True)
    table.add_column("status")
    table.add_column("notes")
    for row in report.transports:
        table.add_row(
            Text(row.name),
            _status_cell(row.status),
            ui.plain(row.note, style=ui.STYLES["ok"] if row.found else ui.STYLES["muted"]),
        )
    return table


def _status_cell(status: str) -> Any:
    """The registry's status string, spelled for whichever console draws the table."""
    return ui.plain(status)


def verdict(report: DoctorReport) -> Any:
    """The line this command exists to produce and never printed: the exit code was the only
    summary it had, and nobody reads an exit code off a screen."""
    if report.ok:
        reason = "the simulator and the scripted pilot run here"
    elif report.missing_core:
        reason = f"a core package is missing: {', '.join(report.missing_core)}"
    elif report.robot and report.robot.error:
        reason = report.robot.error
    elif report.robot and report.robot.probe and report.robot.probe.error:
        reason = report.robot.probe.error
    elif report.robot and report.robot.probe:
        # the probe fails on health OR on a camera that sent no frame, and blaming health
        # for a camera sends the reader to the wrong end of the robot
        bad = [r for r in report.robot.probe.rows if r.state == "fail"]
        reason = f"{bad[0].what}: {bad[0].value}" if bad else "the robot did not report healthy"
    else:
        reason = "the robot did not report healthy"
    counters = [
        f"{sum(1 for e in report.extras if e.ok)}/{len(report.extras)} extras",
        f"{len(report.adapters)} adapters",
        f"{report.bundled_ducks} bundled ducks",
        f"{len(report.assumptions)} unverified assumptions",
    ]
    if report.cloud_keys:
        counters.append(f"no key for {', '.join(report.cloud_keys)}")
    return ui.verdict("success" if report.ok else "failure", reason, counters=counters)


def render(console: Console, report: DoctorReport) -> None:
    """The report as a person reads it: sections in the order you would work through them,
    ending with the one line the exit code used to stand in for."""
    console.print(
        Text.assemble(
            (f"quackd {report.version}", ui.STYLES["key"]),
            (f"  Python {report.python}  {report.platform}", ui.STYLES["muted"]),
            (f"  duck-ipc-proto API v{report.api_version}", ui.STYLES["muted"]),
        )
    )

    _section(console, "core")
    console.print(_checks(report.core))
    console.print(Text(f"  {report.bundled_ducks} bundled ducks", style=ui.STYLES["muted"]))

    _section(console, "providers (every model id: quackd list-models)")
    console.print(_providers_table(report))

    _section(console, "local LLM servers (GET /v1/models, 1.5 s timeout)")
    console.print(_servers_table(report))

    _section(console, "adapters (--robot <adapter>:<backend>)")
    console.print(ui.adapters_table(report.adapters, title=None))

    if report.robot is not None:
        _section(console, report.robot.spec)
        if report.robot.error:
            console.print(ui.fail_line(report.robot.error, hint=_ADAPTER_HINT))
        else:
            console.print(_verbs_table(report.robot))
            if report.robot.probe is not None:
                if report.robot.probe.error:
                    console.print(ui.fail_line(report.robot.probe.error))
                else:
                    console.print(_probe_table(report.robot.probe))
                for advisory in report.robot.probe.advisories:
                    console.print(Text(advisory, style=ui.STYLES["warn"]), soft_wrap=True)

    _section(console, "transports (Microduck backends; --robot microduck:<name>)")
    console.print(_transports_table(report))
    console.print(ui.plain(FLOCK_NOTE, style=ui.STYLES["muted"]))

    _section(console, "optional extras")
    console.print(_checks(report.extras, missing_is_fine=True))

    _section(console, f"upstream assumptions (UNVERIFIED: {len(report.assumptions)})")
    console.print(_assumptions_table(report))
    console.print(_pins_table(report))
    console.print(_read_more(report))

    console.print()
    console.print(verdict(report))


def run_doctor(
    console: Console,
    robot: str | None = None,
    *,
    address: str | None = None,
    camera_url: str | None = None,
    token: str | None = None,
    progress: Progress = None,
) -> bool:
    """Collect, render, and say whether this machine is in a state to run anything."""
    report = collect(robot, address=address, camera_url=camera_url, token=token, progress=progress)
    render(console, report)
    return report.ok
