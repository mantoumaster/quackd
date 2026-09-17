"""The provider protocol: what quackd needs from an LLM and nothing more.

quackd keeps its own vendor-neutral history (`Exchange` = an observation and the decision
it produced). Each provider renders that into its wire format and returns one
`ProviderTurn`. Tools are described once, as JSON Schema, in the Anthropic shape
(`name`, `description`, `input_schema`); other providers translate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

CAMERA_LABEL = "camera {name}:"
"""How a picture is introduced when a body has several. Short on purpose: it sits in front
of every frame of every step, and the observation text already says which is the primary."""


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = ""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    signature: str = ""
    """Opaque and provider-owned, base64 text. Gemini 3 signs every function call it makes
    and refuses the next turn unless the signature is handed back on that same call; other
    providers leave it empty and nothing reads it."""


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    """Tokens spent thinking, when the API counts them apart from the answer (OpenAI does).
    Anthropic folds thinking into `output_tokens`, so it stays 0 there."""

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


class NamedPng(BaseModel):
    """One camera's picture, and which camera took it.

    The name is only ever spoken to a model when a body has more than one camera: a pilot
    told its single view is called `front` would start naming it in sentences nobody needs."""

    model_config = ConfigDict(extra="forbid")

    name: str
    png: bytes


class Observation(BaseModel):
    """What the LLM sees this turn: text, a picture per camera, optionally structured features
    (used by the fake provider and by tests, never rendered to a real model)."""

    model_config = ConfigDict(extra="forbid")

    text: str
    images: list[NamedPng] = Field(default_factory=list)
    """One per camera that gave a frame this step, the primary first. Empty when the body has
    no camera, when the camera gave nothing, or when the model cannot see."""
    cameras: list[str] = Field(default_factory=list)
    """Every camera the body has, answering or not, which is not the same list as `images`.

    Whether a picture is named on the wire is decided from this rather than from how many
    arrived: a two-camera arm whose top lens stalls sends one frame, and sending it bare
    would put the side view under the primary's detections with nothing saying so. Empty for
    a body with one camera or none, which is every request quackd made before an arm could
    have two, so those go out unchanged."""
    features: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str | None = Field(
        default=None, description="Set when this is the result of a tool call."
    )


def name_cameras(obs: Observation) -> bool:
    """Whether this body's pictures are named on the wire. One camera needs no label."""
    return len(obs.cameras) > 1


def labelled(
    images: Sequence[NamedPng],
    image_part: Callable[[bytes], Any],
    text_part: Callable[[str], Any],
    *,
    name_them: bool = False,
) -> list[Any]:
    """A turn's pictures as wire parts, in order, each named when the body has several cameras.

    With a one-camera body this is the single part every provider sent back when a body could
    only have one camera, so a one-camera request goes out unchanged. With several, each
    picture is preceded by a text part naming its camera: two unlabelled images in one message
    are two views of a room with nothing to say which is which.

    `name_them` comes from the body's camera list and not from `len(images)`, because the
    dangerous case is the one where they disagree. A two-camera arm whose primary stalls sends
    exactly one picture, and that picture is the one that most needs saying which lens it is:
    it lands under a detections line measured off the lens that died."""
    if len(images) == 1 and not name_them:
        return [image_part(images[0].png)]
    parts: list[Any] = []
    for image in images:
        parts.append(text_part(CAMERA_LABEL.format(name=image.name)))
        parts.append(image_part(image.png))
    return parts


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call: ToolCall
    text: str | None = None
    raw: Any = Field(
        default=None,
        description="Provider-specific replay payload (Anthropic content blocks incl. thinking).",
    )


class Exchange(BaseModel):
    observation: Observation
    decision: Decision | None = None


class ProviderTurn(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_calls: list[ToolCall] = Field(default_factory=list)
    text: str | None = None
    usage: Usage = Field(default_factory=Usage)
    stop_reason: str | None = None
    raw: Any = None
    thinking: str | None = Field(
        default=None,
        description="What the model reasoned before answering, as the vendor shows it: a "
        "summary on Claude, the reasoning field of an OpenAI-compatible server, Gemini's "
        "thought parts. None when the provider returned nothing of the kind.",
    )


class ProviderError(RuntimeError):
    pass


class ProviderNotInstalled(ProviderError):
    def __init__(self, provider: str, extra: str) -> None:
        super().__init__(
            f"provider {provider!r} needs the optional extra quackd[{extra}] — "
            f'run: uvx --from "quackd[{extra}]" quackd ...  '
            f'or: uv pip install "quackd[{extra}]"'
        )


class ProviderMissingKey(ProviderError):
    def __init__(self, provider: str, env_var: str) -> None:
        super().__init__(
            f"provider {provider!r} needs {env_var} (set it in .env or the environment)"
        )


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str
    supports_vision: bool

    async def step(
        self, system: str, history: list[Exchange], tools: list[dict[str, Any]]
    ) -> ProviderTurn: ...
