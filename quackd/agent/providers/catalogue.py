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
from typing import Any, Literal, get_args

Status = Literal["current", "legacy", "preview", "specialised", "open"]

#: Every status there is, in the order a list of models should be shown in. `list-models` prints
#: them in this order and the browser demo groups its dropdown by it.
STATUSES: tuple[Status, ...] = get_args(Status)


#: When the rates below were last read off the vendor's own pricing page. A price is data and
#: data rots exactly the way the model ids above did, so the date travels with the numbers: it is
#: written into every `run_start`, and a reader deciding whether to trust a dollar figure can see
#: how old the rate that produced it is.
PRICES_CHECKED = "2026-09-21"


@dataclass(frozen=True, slots=True)
class Price:
    """What a vendor charges for one model, in USD per million tokens.

    Four numbers, because that is the shape of every rate card: a full input rate, an output
    rate, and where there is a prompt cache, a cheaper rate for reading one and sometimes a
    dearer one for writing it.

    `None` means the vendor does not publish that rate, NOT that it is free. `pricing.cost_usd`
    charges an unpublished cache rate at the full input rate, which overstates the bill rather
    than understating it: a cost that is too low is the one that gets believed.
    """

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None
    source: str = "catalogue"
    """Where this rate came from, for the record: `catalogue`, `fake`, `self-hosted`,
    `published`, or the name of the flag or variable that overrode it."""

    def record(self) -> dict[str, Any]:
        """The rate as `run_start` and `summary.json` carry it.

        Written into the run rather than looked up at replay, so `quackd trace` prices a run at
        what it cost on the day rather than at whatever this table says months later."""
        return {
            "input": self.input,
            "output": self.output,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "unit": "USD per million tokens",
            "source": self.source,
            # A date for the rates quackd read off a page, and none for a rate somebody
            # handed it: they know where theirs came from, and stamping it with quackd's
            # check date would be quackd vouching for a number it has never seen. The
            # stepper's published rate is re-read whenever this table is, so it shares
            # the date.
            "checked": PRICES_CHECKED if self.source in ("catalogue", "published") else None,
        }


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

    price: Price | None = None
    """What the vendor charges for it, read off the vendor's own page on `PRICES_CHECKED`.

    `None` where no public per-token rate exists, and a run on such a model records
    `cost_usd: null` rather than a zero: a model quackd cannot price is not a free one, and
    printing `$0.00` for a frontier model would be the most expensive kind of wrong."""


# ── the catalogue ───────────────────────────────────────────────────────────────────────
#
# Insertion order is display order, and THE FIRST ENTRY OF EACH VENDOR IS ITS DEFAULT: the model
# `--model` means when it is left off. Writing the default as a separate field would spell the
# same id twice and let the two drift.
#
# `price=` is USD per million tokens, read off the vendor's own pricing page on `PRICES_CHECKED`
# and typed from that page rather than from anybody's recollection. Four rules hold across every
# vendor below, and where one of them bites a particular vendor its own comment says so:
#
# - **The standard on-demand rate.** Batch, priority, provisioned, off-peak and free tiers are
#   ignored: each is a choice the caller makes and quackd cannot see which was made.
# - **The SHORT context band.** Several vendors charge more above a threshold (200k on xAI and
#   the Gemini Pro models, 272k on eight of OpenAI's) and reprice the WHOLE request when it is
#   crossed rather than just the excess. The multiplier is not one number: xAI double both
#   sides, while Gemini and OpenAI double the input and charge 1.5x the output. A long-prompt
#   run is therefore under-costed here, and the vendor comments name the thresholds so a reader
#   knows when to distrust the figure.
# - **The five minute cache-write TTL**, where a vendor sells two. quackd sets no cache at all
#   today, so every cache rate here is one waiting for a caller rather than one in use.
# - **`None` is not free.** Three Cohere models have no published per-token rate and carry no
#   `price` at all; a run on one records `cost_usd: null` and says `cost unpriced`. A model whose
#   vendor genuinely charges nothing is `Price(0.0, 0.0, ...)` and says `$0`.

CATALOGUE: dict[str, tuple[ModelSpec, ...]] = {
    # platform.claude.com models overview and model-deprecations. Claude Mythos is invitation only
    # (Project Glasswing). Opus 4.1 and everything 4.0 or older is retired. Three of these ids are
    # aliases onto a dated snapshot (haiku-4-5, opus-4-5, sonnet-4-5); the rest are the snapshot
    # itself, and appending a date to one of those is an error rather than a pin.
    #
    # Prices: cache writes are the 5 minute TTL rate (1.25x input); the 1 hour rate is 2x input
    # and is not here. Cache reads are a tenth of input on every model except Fable 5.1 and
    # Mythos 5.1, which Anthropic footnote at 0.025x (only the first of those has a row here,
    # Mythos being invitation only), so a Fable 5 read costs four times what a Fable 5.1 read
    # does: do not copy one of those two rows onto the other. Neither pricing page prints an
    # API id at all, so every rate here is mapped from a display label (Claude Opus 5 ->
    # `claude-opus-5`).
    "anthropic": (
        ModelSpec("claude-opus-5", "Claude Opus 5", price=Price(5.0, 25.0, 0.5, 6.25)),
        ModelSpec("claude-fable-5-1", "Claude Fable 5.1", price=Price(10.0, 50.0, 0.25, 12.5)),
        ModelSpec("claude-sonnet-5", "Claude Sonnet 5", price=Price(2.0, 10.0, 0.2, 2.5)),
        ModelSpec("claude-haiku-4-5", "Claude Haiku 4.5", price=Price(1.0, 5.0, 0.1, 1.25)),
        ModelSpec("claude-fable-5", "Claude Fable 5", "legacy", price=Price(10.0, 50.0, 1.0, 12.5)),
        ModelSpec(
            "claude-opus-4-8", "Claude Opus 4.8", "legacy", price=Price(5.0, 25.0, 0.5, 6.25)
        ),
        ModelSpec(
            "claude-opus-4-7", "Claude Opus 4.7", "legacy", price=Price(5.0, 25.0, 0.5, 6.25)
        ),
        ModelSpec(
            "claude-opus-4-6", "Claude Opus 4.6", "legacy", price=Price(5.0, 25.0, 0.5, 6.25)
        ),
        ModelSpec(
            "claude-opus-4-5", "Claude Opus 4.5", "legacy", price=Price(5.0, 25.0, 0.5, 6.25)
        ),
        ModelSpec(
            "claude-sonnet-4-6", "Claude Sonnet 4.6", "legacy", price=Price(3.0, 15.0, 0.3, 3.75)
        ),
        ModelSpec(
            "claude-sonnet-4-5", "Claude Sonnet 4.5", "legacy", price=Price(3.0, 15.0, 0.3, 3.75)
        ),
    ),
    # developers.openai.com models and deprecations. The whole gpt-5 and o-series generation has
    # shutdown dates in 2026, so none of it is here. The Daybreak, Cyber and Rosalind models need
    # their own approval. `gpt-oss-*` are weights, not an endpoint.
    #
    # Prices: the short-context band. A prompt over 272k tokens is billed at 2x input and 1.5x
    # output for the WHOLE request, and nothing here sees that coming. That band is not only
    # the flagships: the page carries long-context columns for eight of these ids, the four
    # 5.6-and-later models plus gpt-5.5, gpt-5.4 and both Pro tiers. Cache writes are 1.25x
    # input on GPT-5.6 and later and genuinely nothing before it, which is why half this list
    # carries a zero there and half does not. gpt-5.6-sol's rate is promotional and holds
    # only into late 2026.
    "openai": (
        ModelSpec("gpt-5.6-sol", "GPT-5.6 Sol", price=Price(4.0, 20.0, 0.4, 5.0)),
        ModelSpec(
            "gpt-6-astra", "GPT-6 Astra", api="responses", price=Price(10.0, 50.0, 1.0, 12.5)
        ),
        ModelSpec("gpt-5.6-terra", "GPT-5.6 Terra", price=Price(2.0, 12.0, 0.2, 2.5)),
        ModelSpec("gpt-5.6-luna", "GPT-5.6 Luna", price=Price(0.2, 1.2, 0.02, 0.25)),
        ModelSpec("gpt-5.5", "GPT-5.5", "legacy", price=Price(5.0, 30.0, 0.5, 0.0)),
        ModelSpec("gpt-5.4", "GPT-5.4", "legacy", price=Price(2.5, 15.0, 0.25, 0.0)),
        ModelSpec("gpt-5.4-mini", "GPT-5.4 mini", "legacy", price=Price(0.75, 4.5, 0.075, 0.0)),
        ModelSpec("gpt-5.4-nano", "GPT-5.4 nano", "legacy", price=Price(0.2, 1.25, 0.02, 0.0)),
        ModelSpec("gpt-5.2", "GPT-5.2", "legacy", price=Price(1.75, 14.0, 0.175, 0.0)),
        ModelSpec("gpt-5.1", "GPT-5.1", "legacy", price=Price(1.25, 10.0, 0.125, 0.0)),
        ModelSpec("gpt-4.1", "GPT-4.1", "legacy", price=Price(2.0, 8.0, 0.5, 0.0)),
        ModelSpec("gpt-4.1-mini", "GPT-4.1 mini", "legacy", price=Price(0.4, 1.6, 0.1, 0.0)),
        ModelSpec("gpt-4o", "GPT-4o", "legacy", price=Price(2.5, 10.0, 1.25, 0.0)),
        ModelSpec("gpt-4o-mini", "GPT-4o mini", "legacy", price=Price(0.15, 0.6, 0.075, 0.0)),
        ModelSpec(
            "gpt-5.5-pro", "GPT-5.5 Pro", "specialised", api="responses", price=Price(30.0, 180.0)
        ),
        ModelSpec(
            "gpt-5.4-pro", "GPT-5.4 Pro", "specialised", api="responses", price=Price(30.0, 180.0)
        ),
        ModelSpec(
            "gpt-5.2-pro", "GPT-5.2 Pro", "specialised", api="responses", price=Price(21.0, 168.0)
        ),
        ModelSpec(
            "gpt-5.3-codex",
            "GPT-5.3 Codex",
            "specialised",
            api="responses",
            price=Price(1.75, 14.0, 0.175, 0.0),
        ),
        ModelSpec(
            "chat-latest",
            "ChatGPT Instant (moves with ChatGPT)",
            "specialised",
            price=Price(5.0, 30.0, 0.5),
        ),
    ),
    # ai.google.dev models and deprecations. Gemini 3.1 Flash-Lite has a shutdown date already,
    # the 2.0 family and the Gemini 3 Pro preview are gone, and Robotics ER 1.6 went on
    # 2026-08-31. ER 2 is here because quackd drives robots and that is its subject. Google's own
    # page calls 3.5 Flash "our legacy Flash model", so that is where it sits.
    #
    # Prices: three Flash models and Robotics ER 2 are on an introductory rate that DOUBLES on
    # 2027-01-01, which the page says in as many words. After that date those four under-report
    # by half until somebody re-reads it. `gemini-3.1-pro-preview` and `gemini-2.5-pro` are the
    # short band; above 200k their input and cache rates double and their output goes to 1.5x,
    # and the rest are flat. Google charge nothing to create a cache, billing storage by the
    # hour instead, which is not a per-token rate and is not costed here. Flash-Lite is the one
    # model here with no published cached-input rate, so a cached prompt on it is charged at
    # the full input rate like every other unpublished cache rate in this file.
    "gemini": (
        ModelSpec("gemini-3.8-flash", "Gemini 3.8 Flash", price=Price(0.75, 3.75, 0.075, 0.0)),
        ModelSpec("gemini-3.7-flash", "Gemini 3.7 Flash", price=Price(0.75, 3.75, 0.075, 0.0)),
        ModelSpec("gemini-3.6-flash", "Gemini 3.6 Flash", price=Price(0.75, 3.75, 0.075, 0.0)),
        ModelSpec("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", price=Price(0.3, 2.5)),
        ModelSpec(
            "gemini-3.1-pro-preview",
            "Gemini 3.1 Pro (preview)",
            "preview",
            price=Price(2.0, 12.0, 0.2, 0.0),
        ),
        ModelSpec(
            "gemini-3-flash-preview",
            "Gemini 3 Flash (preview)",
            "preview",
            price=Price(0.5, 3.0, 0.05, 0.0),
        ),
        ModelSpec(
            "gemini-robotics-er-2-preview",
            "Gemini Robotics ER 2 (preview)",
            "preview",
            price=Price(1.0, 5.0, 0.1, 0.0),
        ),
        ModelSpec(
            "gemini-3.5-flash", "Gemini 3.5 Flash", "legacy", price=Price(1.5, 9.0, 0.15, 0.0)
        ),
        ModelSpec(
            "gemini-2.5-pro", "Gemini 2.5 Pro", "legacy", price=Price(1.25, 10.0, 0.125, 0.0)
        ),
        ModelSpec(
            "gemini-2.5-flash", "Gemini 2.5 Flash", "legacy", price=Price(0.3, 2.5, 0.03, 0.0)
        ),
        ModelSpec(
            "gemini-2.5-flash-lite",
            "Gemini 2.5 Flash-Lite",
            "legacy",
            price=Price(0.1, 0.4, 0.01, 0.0),
        ),
    ),
    # docs.x.ai. grok-4 and the whole grok-3 and grok-4-fast line were retired on 2026-05-15 and
    # now answer as grok-4.3 without saying so, which is the kind of silence a catalogue exists
    # to stop. All seven take an image and all seven do function calling.
    #
    # Prices: the band below 200k prompt tokens. Above it xAI charge 2x for the whole request.
    # They publish a cached-input rate and say nothing whatever about cache creation, so
    # `cache_write` is None rather than 0 and any such tokens are charged at the full rate.
    "grok": (
        ModelSpec("grok-4.6", "Grok 4.6", price=Price(2.0, 6.0, 0.5)),
        ModelSpec("grok-4.5", "Grok 4.5", price=Price(2.0, 6.0, 0.3)),
        ModelSpec("grok-4.3", "Grok 4.3", price=Price(1.25, 2.5, 0.2)),
        ModelSpec(
            "grok-4.20-0309-reasoning",
            "Grok 4.20 (reasoning)",
            "legacy",
            price=Price(1.25, 2.5, 0.2),
        ),
        ModelSpec(
            "grok-4.20-0309-non-reasoning",
            "Grok 4.20 (non-reasoning)",
            "legacy",
            price=Price(1.25, 2.5, 0.2),
        ),
        ModelSpec(
            "grok-4.20-multi-agent-0309",
            "Grok 4.20 multi-agent",
            "specialised",
            price=Price(1.25, 2.5, 0.2),
        ),
        ModelSpec("grok-build-0.1", "Grok Build 0.1", "specialised", price=Price(1.0, 2.0, 0.2)),
    ),
    # docs.mistral.ai. Magistral retired on 2026-07-31, and Devstral and Pixtral are deprecated.
    # Medium 3.5 is the one id here that does not follow the dated pattern: Mistral's lifecycle
    # page has moved to name-major-minor and the model card lists `mistral-medium-3-5`. The last
    # two are models Mistral serves but did not train, at its own address under its own ids.
    #
    # Prices: the standard tier, with cached input a tenth of input across the board. Batch is
    # half and priority is 1.75x, and neither is here. Leanstral is listed at Free with no end
    # date attached, which is a real zero rather than a missing rate, and also the kind of zero
    # that can stop being one without an announcement.
    "mistral": (
        ModelSpec("mistral-medium-3-5", "Mistral Medium 3.5", price=Price(1.5, 7.5, 0.15, 0.0)),
        ModelSpec("mistral-large-2512", "Mistral Large 3", price=Price(0.5, 1.5, 0.05, 0.0)),
        ModelSpec("mistral-small-2603", "Mistral Small 4", price=Price(0.15, 0.6, 0.015, 0.0)),
        ModelSpec(
            "ministral-14b-2512", "Ministral 3 14B", "open", price=Price(0.2, 0.2, 0.02, 0.0)
        ),
        ModelSpec(
            "ministral-8b-2512", "Ministral 3 8B", "open", price=Price(0.15, 0.15, 0.015, 0.0)
        ),
        ModelSpec("ministral-3b-2512", "Ministral 3 3B", "open", price=Price(0.1, 0.1, 0.01, 0.0)),
        ModelSpec(
            "codestral-2508",
            "Codestral 25.08",
            "specialised",
            vision=False,
            price=Price(0.3, 0.9, 0.03, 0.0),
        ),
        ModelSpec(
            "zai-glm-5-2",
            "Z.ai GLM 5.2 (hosted by Mistral)",
            "preview",
            vision=False,
            price=Price(1.4, 4.4, 0.14, 0.0),
        ),
        ModelSpec(
            "labs-leanstral-1-5",
            "Leanstral 1.5 (Lean 4 proofs, Mistral labs)",
            "preview",
            vision=False,
            price=Price(0.0, 0.0, 0.0, 0.0),
        ),
    ),
    # api-docs.deepseek.com. `deepseek-chat` and `deepseek-reasoner` are retired aliases.
    # Prices are the PEAK band. DeepSeek halve everything outside 01:00-04:00 and 06:00-10:00
    # UTC on weekdays, so an off-peak run costs half of what this says. `input` is their
    # cache-miss price and `cache_read` their cache-hit one; creation is not charged.
    "deepseek": (
        ModelSpec("deepseek-flash", "DeepSeek V4.1 Flash", price=Price(0.3, 1.2, 0.006, 0.0)),
        ModelSpec(
            "deepseek-v4-pro", "DeepSeek V4 Pro", vision=False, price=Price(1.32, 3.96, 0.044, 0.0)
        ),
    ),
    # docs.cohere.com, reached through the OpenAI compatibility endpoint. Command A Translate and
    # Command A Vision are out for the same reason, which their own pages state: no tool use. The
    # North models are out because nothing in their docs says they have any either.
    #
    # Prices: THE COMMAND A FAMILY HAS NO PER-TOKEN RATE, the default included. Every
    # per-token generative price on Cohere's page belongs to the older Command R line, which is
    # where the three rates below come from. Command A and Command A Reasoning are not on that
    # page at all and are sold as dedicated instances by the hour, which cannot be turned into
    # a per-token figure. Command A+ is on it, but as free open weights, with the page's own
    # FAQ saying a production key is billed pay as you go at a rate it does not state: a zero
    # that is true of a trial key and false of the calls quackd would make, so it is left
    # unpriced rather than recorded as free. A default Cohere run therefore reports
    # `cost_usd: null` and says `cost unpriced`, and `--price` is how you tell quackd your own
    # rate. This is the case the null was built for.
    "cohere": (
        ModelSpec("command-a-plus-05-2026", "Command A+"),
        ModelSpec("command-a-03-2025", "Command A", vision=False),
        ModelSpec(
            "command-a-reasoning-08-2025", "Command A Reasoning", "specialised", vision=False
        ),
        ModelSpec(
            "command-r-plus-08-2024", "Command R+", "legacy", vision=False, price=Price(2.5, 10.0)
        ),
        ModelSpec("command-r-08-2024", "Command R", "legacy", vision=False, price=Price(0.15, 0.6)),
        ModelSpec(
            "command-r7b-12-2024", "Command R7B", "legacy", vision=False, price=Price(0.0375, 0.15)
        ),
    ),
    # Alibaba Cloud Model Studio. Only the rolling ids are here: the dated snapshots behind them
    # (`qwen3.8-max-0902` and the rest) are a pinning mechanism, not a menu, and Alibaba gives them
    # a month of notice against three for a rolling id. Qwen3.7 Max sits under the vendor's own
    # "Legacy models" heading even though 3.7 Plus and Flash do not.
    #
    # Prices: the Singapore international catalogue at the FIRST input-length tier, which is the
    # sharpest under-report in this file. `qwen3-coder-plus` goes from $1/$5 to $6/$60 above
    # 256k, twelve times the output rate, and the China region is priced differently again.
    # Where a model splits its output rate by thinking mode this is the non-thinking one
    # (`qwen-plus` is $1.2 against $4 thinking). Alibaba publish no per-model cache rate at all,
    # only a percentage rule that differs between implicit and explicit caching, so every cache
    # field here is None.
    "qwen": (
        ModelSpec("qwen3.8-max", "Qwen3.8 Max", price=Price(2.0, 6.0)),
        ModelSpec("qwen3.8-flash", "Qwen3.8 Flash", price=Price(0.15, 0.47)),
        ModelSpec("qwen3.7-plus", "Qwen3.7 Plus", price=Price(0.4, 1.6)),
        ModelSpec("qwen3.7-flash", "Qwen3.7 Flash", price=Price(0.03, 0.13)),
        ModelSpec("qwen3.7-max", "Qwen3.7 Max", "legacy", vision=False, price=Price(2.5, 7.5)),
        ModelSpec("qwen3.6-plus", "Qwen3.6 Plus", "legacy", price=Price(0.5, 3.0)),
        ModelSpec("qwen3.6-flash", "Qwen3.6 Flash", "legacy", price=Price(0.25, 1.5)),
        ModelSpec("qwen3.5-plus", "Qwen3.5 Plus", "legacy", price=Price(0.4, 2.4)),
        ModelSpec("qwen3.5-flash", "Qwen3.5 Flash", "legacy", price=Price(0.1, 0.4)),
        ModelSpec("qwen3-max", "Qwen3 Max", "legacy", vision=False, price=Price(1.2, 6.0)),
        ModelSpec("qwen-plus", "Qwen Plus", "legacy", vision=False, price=Price(0.4, 1.2)),
        ModelSpec("qwen-flash", "Qwen Flash", "legacy", vision=False, price=Price(0.05, 0.4)),
        ModelSpec(
            "qwen3-coder-next",
            "Qwen3 Coder Next",
            "specialised",
            vision=False,
            price=Price(0.3, 1.5),
        ),
        ModelSpec(
            "qwen3-coder-plus",
            "Qwen3 Coder Plus",
            "specialised",
            vision=False,
            price=Price(1.0, 5.0),
        ),
        ModelSpec(
            "qwen3-coder-flash",
            "Qwen3 Coder Flash",
            "specialised",
            vision=False,
            price=Price(0.3, 1.5),
        ),
        ModelSpec("qwen3.8-27b", "Qwen3.8 27B", "open", price=Price(0.5, 3.0)),
        ModelSpec(
            "qwen3.8-2.4t-a95b", "Qwen3.8 2.4T-A95B", "open", vision=False, price=Price(2.0, 6.0)
        ),
        ModelSpec("qwen3.6-35b-a3b", "Qwen3.6 35B-A3B", "open", price=Price(0.375, 2.25)),
        ModelSpec("qwen3.6-27b", "Qwen3.6 27B", "open", price=Price(0.6, 3.6)),
        ModelSpec("qwen3.5-397b-a17b", "Qwen3.5 397B-A17B", "open", price=Price(0.6, 3.6)),
        ModelSpec("qwen3.5-122b-a10b", "Qwen3.5 122B-A10B", "open", price=Price(0.4, 3.2)),
        ModelSpec("qwen3.5-27b", "Qwen3.5 27B", "open", price=Price(0.3, 2.4)),
        ModelSpec("qwen3.5-35b-a3b", "Qwen3.5 35B-A3B", "open", price=Price(0.25, 2.0)),
    ),
    # platform.kimi.ai. Everything before K2.6 was discontinued on 2026-05-25 and answers 404.
    # Only K3 accepts a forced tool call: the other three have thinking permanently on, and that
    # narrows tool_choice to auto or none, which is why `kimi.py` asks rather than insists.
    #
    # Prices: `input` is cache-miss and `cache_read` cache-hit. Only K3 publishes a cache-write
    # rate, at the 5 minute TTL; the K2 table has no such column, so those three are None.
    "kimi": (
        ModelSpec("kimi-k3", "Kimi K3", price=Price(3.0, 15.0, 0.3, 3.0)),
        ModelSpec("kimi-k2.6", "Kimi K2.6", price=Price(0.95, 4.0, 0.16)),
        ModelSpec("kimi-k2.7-code", "Kimi K2.7 Code", "specialised", price=Price(0.95, 4.0, 0.19)),
        ModelSpec(
            "kimi-k2.7-code-highspeed",
            "Kimi K2.7 Code (high speed)",
            "specialised",
            price=Price(1.9, 8.0, 0.38),
        ),
    ),
    # docs.z.ai. The `v` models are the vision line and the rest are text only. GLM-4.5V is not
    # here: it is the one of those whose page does not give it function calling.
    #
    # Prices are flat, with no context bands on the page today, though earlier GLM generations
    # had them and a refresh should look again. Three Flash models are genuinely free. No
    # cache-creation rate is published for any of them, and the "Cached Input Storage" column
    # reads "Limited-time Free", which is a storage promotion rather than a write rate and is
    # deliberately not mapped onto one. `glm-4-32b-0414-128k` has no cached-input rate either,
    # so its cache reads are charged at the full input rate.
    "glm": (
        ModelSpec("glm-5.3", "GLM-5.3", vision=False, price=Price(1.4, 4.4, 0.26)),
        ModelSpec("glm-5.3-flash", "GLM-5.3 Flash", price=Price(0.15, 0.5, 0.03)),
        ModelSpec("glm-5.2", "GLM-5.2", "legacy", vision=False, price=Price(1.4, 4.4, 0.26)),
        ModelSpec("glm-5.1", "GLM-5.1", "legacy", vision=False, price=Price(1.4, 4.4, 0.26)),
        ModelSpec("glm-5", "GLM-5", "legacy", vision=False, price=Price(1.0, 3.2, 0.2)),
        ModelSpec("glm-4.7", "GLM-4.7", "legacy", vision=False, price=Price(0.6, 2.2, 0.11)),
        ModelSpec(
            "glm-4.7-flash",
            "GLM-4.7 Flash",
            "legacy",
            vision=False,
            price=Price(0.0, 0.0, 0.0, 0.0),
        ),
        ModelSpec(
            "glm-4.7-flashx", "GLM-4.7 FlashX", "legacy", vision=False, price=Price(0.07, 0.4, 0.01)
        ),
        ModelSpec("glm-4.6", "GLM-4.6", "legacy", vision=False, price=Price(0.6, 2.2, 0.11)),
        ModelSpec("glm-4.5", "GLM-4.5", "legacy", vision=False, price=Price(0.6, 2.2, 0.11)),
        ModelSpec("glm-4.5-x", "GLM-4.5 X", "legacy", vision=False, price=Price(2.2, 8.9, 0.45)),
        ModelSpec(
            "glm-4.5-air", "GLM-4.5 Air", "legacy", vision=False, price=Price(0.2, 1.1, 0.03)
        ),
        ModelSpec(
            "glm-4.5-airx", "GLM-4.5 AirX", "legacy", vision=False, price=Price(1.1, 4.5, 0.22)
        ),
        ModelSpec(
            "glm-4.5-flash",
            "GLM-4.5 Flash",
            "legacy",
            vision=False,
            price=Price(0.0, 0.0, 0.0, 0.0),
        ),
        ModelSpec(
            "glm-4-32b-0414-128k", "GLM-4 32B (128k)", "legacy", vision=False, price=Price(0.1, 0.1)
        ),
        ModelSpec("glm-4.6v", "GLM-4.6V", "specialised", price=Price(0.3, 0.9, 0.05)),
        ModelSpec(
            "glm-4.6v-flash", "GLM-4.6V Flash", "specialised", price=Price(0.0, 0.0, 0.0, 0.0)
        ),
        ModelSpec(
            "glm-4.6v-flashx", "GLM-4.6V FlashX", "specialised", price=Price(0.04, 0.4, 0.004)
        ),
    ),
    # Meta retired the hosted Llama API in July 2026. What replaced it is the Meta Model API,
    # which is OpenAI shaped and serves Muse Spark. The contributor tier is cheaper because Meta
    # trains on what you send it, which is why the label says so rather than the release notes.
    # Meta price per TIER rather than per model, so 1.1 costs exactly what 1.3 does; if they
    # ever split those rows these constants go stale in silence. There is no long-context
    # premium on the 1M window, which is unusual enough to be worth writing down.
    "meta": (
        ModelSpec("muse-spark-1.3", "Muse Spark 1.3", price=Price(1.25, 4.25, 0.15)),
        ModelSpec("muse-spark-1.2", "Muse Spark 1.2", "legacy", price=Price(1.25, 4.25, 0.15)),
        ModelSpec("muse-spark-1.1", "Muse Spark 1.1", "legacy", price=Price(1.25, 4.25, 0.15)),
        ModelSpec(
            "muse-spark-1.3-contributor",
            "Muse Spark 1.3 (contributor tier, Meta trains on your prompts)",
            "specialised",
            price=Price(0.1, 0.2, 0.002),
        ),
        ModelSpec(
            "muse-spark-1.2-contributor",
            "Muse Spark 1.2 (contributor tier, Meta trains on your prompts)",
            "specialised",
            price=Price(0.1, 0.2, 0.002),
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
