"""The policy loop: one segment of a learned policy driving the arm, paced on the arm's clock.

A segment is what `pick` and `manipulate` hand the arm to. It runs as the backend's policy task
(`LeRobotReal._run_policy`, which takes the segment's first reading, refuses what it must and
sets the step cap), and this loop is its body: a tick at a time, it reads the arm, judges the
reading, takes what the runner has answered, asks it for more, and sends one goal.

**The rate** is the runner's, from the source it names (`Features`), and a rate that is not a
finite number inside `MIN_RATE_HZ`..`MAX_RATE_HZ` refuses the segment. quackd never measures
one: a rate is a fact about the data a policy learned from, and the loop's own pace is reported,
never used to choose it.

**The pace** is the clock's. Tick `k` is due at `start + k * period`, from the integer `k` and
never from a sum of sleeps, so a tick that ran long costs its own tick and not every one after
it. A tick that overruns the next deadline skips to the next whole period after the clock's now,
counts the ticks it missed, and never sends twice in one period. A tick never starts before its
deadline either: the simulator's clock wakes on its own steps, the nearest to what was asked,
and a period that is not a whole number of steps would otherwise wake a tick in the period
before its own (`PolicyLoop._pace`).

**The speed cap** is the verbs' own, `max_step_deg / TICK_S` degrees a second, so the one
setting (`QUACKD_LEROBOT_MAX_STEP_DEG`) governs both. Per send it is that speed over the pacer's
rate, clamped into `(0, max_step_deg]` (`speed_cap`), and the backend writes it on the follower's
config for the length of the segment.

**Chunks.** A request is stamped with the tick it was read at, and the chunk that answers it
holds one action per tick from that tick. When it arrives, the actions for ticks already played
are dropped and the rest replaces the queue's tail, never appended to it, because the newer
chunk saw a newer arm. At most one request is outstanding. A tick with nothing queued sends
nothing (the arm holds its last goal), counts as starved, and `STARVE_S` of that ends the
segment, with `FIRST_CHUNK_S` of grace for a segment's first chunk. Each segment has an epoch,
and a chunk from an earlier one is thrown away.

**The runner's thread.** Every call on the runner goes to a single worker of its own, never to
the default pool the bus's calls run in (`LeRobotReal._call`), so a runner that never answers
cannot hold up a read, a hold or a heartbeat. A segment's reset waits for the one before it,
behind whatever inference is still on that worker, for no longer than `RESET_S`.

**Lockstep.** On the simulator (a clock whose `lockstep` is true) time stands still while
nothing sleeps on it, so a runner's inference is awaited in the loop's own turn and costs no sim
time. Its answer is then held back `k` ticks, `k` from the runner's declared latency, while the
pacer goes on ticking and playing the old queue, and is judged only when it is let go: that is
where it would have landed on the arm, a chunk thrown away and an error raised included.
A runner that answers at once (`k` of 0) is awaited in its own tick on either clock, which is
how a `PolicyLike` has always run: one `act` a tick, its goal sent that tick.

**Tick mode** (`Features.per_tick`): a runner that carries state from tick to tick is asked
every tick, and a tick it never saw, skipped by the pacer or not answered by the next, ends the
segment with the arm held and resets the runner. Its delay line is a tick deep at most: it is
asked again only once it has answered, so one that declares more than a tick to answer could
never keep up on the arm, and is refused on either clock rather than rehearsed on the
simulator's.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import math
import numbers
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from quackd_lerobot.policy.runner import Chunk, Features, Observation, PolicyRunner
from quackd_lerobot.verbs import (
    JOINTS,
    PICK_SETTLE_S,
    STALL_DEG,
    TICK_S,
    SegmentEnd,
    SegmentHow,
    SegmentStats,
)

if TYPE_CHECKING:
    from quackd_lerobot.real import LeRobotReal

T = TypeVar("T")

REGISTER_PERIOD_S = 0.5
"""How often a policy segment reads torque and temperature, in the transport's time. Every tick
reads the joints, which is what the policy and `holding` need; the two registers are two more
bus transactions each, and neither changes in a tenth of a second. A servo heats over seconds
and one that trips its overload drops torque for good, so twice a second catches both in time
to hold the arm, and the heartbeat's own probe was reading them at that rate anyway."""
FAILED_SENDS = 3
"""How many sends in a row may fail before a policy segment ends. A Feetech bus loses the odd
status packet, which is no reason to take the arm off a policy mid grasp; three in a row is a
bus or an arm that has stopped taking goals, and the segment ends there, holding the arm."""
CLIP_SUSTAIN_S = 1.0
"""How long a joint's goal may stay clipped to its travel before a policy segment ends, in the
transport's time. A policy goal past the travel is clipped and counted rather than refused
(ADR-0036), because one a tick over should not abort a grasp. A second of them in a row is a
policy pushing the arm somewhere it cannot go, and the segment ends on it."""

MIN_RATE_HZ = 1.0
MAX_RATE_HZ = 60.0
"""The rates quackd paces a policy at. quackd's own bounds, and no measurement: a rate outside
them is more likely a wrong number than a policy, a period given where a frequency was meant,
and above the ceiling a tick is shorter than a read and a send of the arm's bus are likely to
take together, and than a few steps of the simulator's clock. Below the floor every send is a
step the speed cap clamps anyway."""
STARVE_S = 1.0
"""How long, in the transport's time, a segment may have nothing to send before it ends. The
arm holds its last goal meanwhile, which is safe and is also a policy that has stopped driving,
and a second of it is a runner that has fallen behind the arm rather than one that is slow."""
FIRST_CHUNK_S = 5.0
"""The grace a segment's first chunk gets, in place of `STARVE_S`: the first inference after a
reset is the slow one, with caches and buffers built on it, and nothing has been sent yet, so
the arm is exactly where the segment found it while it waits."""
RESET_S = 5.0
"""How long, on the wall's clock, a segment's start waits for the runner's reset, which queues
behind any inference the last segment left on the runner's worker. A runner still busy after
that is not handed the arm, and the segment is refused rather than started on a stale policy."""
STALL_S = 1.0
"""How long every joint of the arm may read within `STALL_DEG` of where it was while the policy
is sending, before `manipulate`'s segment ends on it, with the arm held: an arm that has stopped
moving under a policy that is still sending is a policy that has done what it will do, or one
pressing the arm on something it cannot move, whose last goal must not be left pushing a servo.
Only ticks that sent count, so a policy starved of chunks is starving and not stalled."""
REFILL_SHARE = 0.5
"""A chunked runner is asked again once what is left of its last chunk is this share of it or
less: early enough that the next chunk lands before the queue runs dry, late enough that most of
each chunk is played."""
WORKER = "quackd-policy"
"""The name of the runner's own worker thread, which is never one of the bus's."""
EXACT = 1e-9
"""How near a whole number of ticks a declared latency has to be to be that number, and how near
its deadline a clock has to read for a tick to be due, allowing for the float arithmetic of a
sum and for nothing else."""


@dataclass(frozen=True)
class Segment:
    """What one `do` asked for: `verb` is `pick` or `manipulate`, `instruction` what the runner
    is told, `max_s` the segment's time on the transport's clock and `max_chunks` how many chunks
    it may play, each None for no limit."""

    verb: str
    instruction: str
    max_s: float | None = None
    max_chunks: int | None = None

    @property
    def ends_on_holding(self) -> bool:
        """`pick` is done when something is held, judged on the loop's own reads each tick."""
        return self.verb == "pick"

    @property
    def ends_on_stall(self) -> bool:
        """`manipulate` is done when the arm stops moving under it (`STALL_S`)."""
        return self.verb == "manipulate"


@dataclass(frozen=True)
class Plan:
    """What a segment's start settled: the runner's features, the pacer's period, the step cap
    per send and the ticks a chunk is held back on a lockstep clock."""

    features: Features
    period_s: float
    cap_deg: float
    latency_ticks: int


def speed_cap(max_step_deg: float, rate_hz: float) -> float:
    """The step cap per send for a policy at `rate_hz`: the verbs' speed, `max_step_deg` per
    `TICK_S`, over the pacer's rate, clamped into `(0, max_step_deg]`. A policy at the verbs' own
    rate gets the verbs' own cap, a faster one a smaller step and the same speed, and a slower one
    never more than one verb step. A rate that gives no positive finite step gives none at all."""
    per_send = max_step_deg / TICK_S / rate_hz if rate_hz else math.nan
    if not (math.isfinite(per_send) and per_send > 0):
        raise ValueError(f"no step cap for {max_step_deg!r} degrees at {rate_hz!r} Hz")
    return min(float(max_step_deg), per_send)


def _finite(value: object) -> bool:
    """A finite real number, whatever library made it (a numpy float is one), and never a bool,
    which Python counts as a number and no runner means as one."""
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def rate_refusal(features: Features) -> str | None:
    """Why a runner's declared rate cannot pace a segment, or None."""
    rate = features.rate_hz
    if _finite(rate) and MIN_RATE_HZ <= float(rate) <= MAX_RATE_HZ:
        return None
    return (
        f"the policy was not started: it declares {rate!r} Hz ({features.rate_source}), and "
        f"quackd paces a policy at {MIN_RATE_HZ:g} to {MAX_RATE_HZ:g} Hz. Give it the rate of "
        "the data it was trained on"
    )


def latency_ticks(latency_s: float, rate_hz: float) -> int:
    """How many ticks a chunk is held back on a lockstep clock: the declared latency in ticks,
    rounded up, since a chunk cannot be played before it would have arrived."""
    return max(0, math.ceil(latency_s * rate_hz - EXACT))


def clipped_too_long(
    goals: Mapping[str, float],
    since: dict[str, float],
    now: float,
    travel: Mapping[str, tuple[float, float]],
) -> str | None:
    """Keep `since`, when each joint's goal first went past its travel in the run of ticks it
    has stayed there, and say which has stayed there `CLIP_SUSTAIN_S`. A tick whose action
    leaves a joint out keeps its run going."""
    for joint, goal in goals.items():
        span = travel.get(joint)
        if span is not None and not span[0] <= goal <= span[1]:
            since.setdefault(joint, now)
        else:
            since.pop(joint, None)
    stuck = sorted(joint for joint, at in since.items() if now - at >= CLIP_SUSTAIN_S)
    if not stuck:
        return None
    return (
        f"the policy's goal for {', '.join(stuck)} stayed past the travel for "
        f"{CLIP_SUSTAIN_S:g} s, clipped to its end on every tick, so it was driving the arm "
        "somewhere it cannot go"
    )


@dataclass
class _Request:
    """One request to the runner: the segment's epoch, the tick it was read at, and the call on
    the runner's worker."""

    epoch: int
    tick: int
    future: concurrent.futures.Future[Chunk]


@dataclass
class _Run:
    """One segment's working state, kept apart from the loop, which outlives it."""

    started: float
    registers_at: float
    clips_at: int
    queue: deque[tuple[int, Mapping[str, Any]]] = field(default_factory=deque)
    held_back: deque[tuple[int, _Request]] = field(default_factory=deque)
    """Requests a lockstep clock has had answered and holds back until their tick, oldest
    first. An answer is judged only when it is let go, so one thrown away and an error raised
    land where the arm would first have met them."""
    last_len: int = 0
    got_first: bool = False
    done: bool = False
    done_at: float | None = None
    starved_since: float | None = None
    still: tuple[float, dict[str, float]] | None = None
    last_sent: dict[str, float] = field(default_factory=dict)
    clipped_since: dict[str, float] = field(default_factory=dict)
    failed_sends: int = 0
    ticks: int = 0
    skipped: int = 0
    chunks: int = 0
    starved: int = 0
    stale: int = 0
    late: int = 0


class PolicyLoop:
    """The loop one runner's segments run in, kept by the backend for as long as it has that
    runner, so that its epoch, its worker and a request an earlier segment left in flight carry
    from one segment to the next.

    `starve_s`, `first_chunk_s` and `reset_s` are `STARVE_S`, `FIRST_CHUNK_S` and `RESET_S`,
    as attributes a test can shorten."""

    def __init__(self, runner: PolicyRunner) -> None:
        self.runner = runner
        self.epoch = 0
        """The segment the loop is on, counted from 1. A chunk asked for under another is
        thrown away when it arrives."""
        self.starve_s = STARVE_S
        self.first_chunk_s = FIRST_CHUNK_S
        self.reset_s = RESET_S
        self.instruction = ""
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._inflight: _Request | None = None

    # ── the runner's own thread ─────────────────────────────────────────────────────────

    def _submit(self, fn: Callable[..., T], *args: Any) -> concurrent.futures.Future[T]:
        """Run `fn` on the runner's single worker, made the first time it is needed."""
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=WORKER
            )
        return self._executor.submit(fn, *args)

    @staticmethod
    async def _answered(future: concurrent.futures.Future[Any], within: float | None) -> bool:
        """Wait for `future`, for at most `within` seconds of the event loop's clock or for as
        long as it takes. Whatever ends the wait, a timeout or a cancellation of the caller,
        lets go of it here: a call still queued is cancelled, one already running goes on in
        its thread, and nothing waits on it any more. A runner's error is read off `future`
        where its answer is judged, and this copy of it is marked read, or asyncio would log it
        as never retrieved when it is collected."""
        wrapped = asyncio.wrap_future(future)
        try:
            done, _ = await asyncio.wait({wrapped}, timeout=within)
        finally:
            if not wrapped.done():
                wrapped.cancel()
            elif not wrapped.cancelled():
                wrapped.exception()
        return bool(done)

    async def _ask(self, fn: Callable[..., T], *args: Any) -> T | str:
        """A call on the runner at a segment's start, within `reset_s`, or why it failed."""
        future = self._submit(fn, *args)
        name = getattr(fn, "__name__", "call")
        if not await self._answered(future, self.reset_s):
            return (
                f"the policy was not started: its {name} had not come back after "
                f"{self.reset_s:g} s, behind whatever the last segment left it thinking about, "
                "so quackd will not hand it the arm"
            )
        if future.cancelled():
            return f"the policy was not started: its {name} was cancelled"
        if (error := future.exception()) is not None:
            return f"the policy was not started: its {name} raised {type(error).__name__}: {error}"
        return future.result()

    async def call(self, fn: Callable[..., T], *args: Any, within: float | None = None) -> T:
        """`fn(*args)` on the runner's own worker, behind anything it is doing, for a caller
        outside a segment: the connect asking whether the policy fits the arm. It raises what
        `fn` raised, and a TimeoutError once `within` seconds, `reset_s` unless it is given,
        have passed without an answer."""
        wait = self.reset_s if within is None else within
        future = self._submit(fn, *args)
        if not await self._answered(future, wait):
            name = getattr(fn, "__name__", "call")
            raise TimeoutError(f"the policy's {name} had not come back after {wait:g} s")
        return future.result()

    def close(self) -> None:
        """Close the runner on its own worker, after whatever it is doing, without waiting for
        it. The worker is kept, so a segment after a reconnect still queues behind anything the
        runner was left doing and never runs beside it."""
        if self._executor is not None:
            self._executor.submit(self.runner.close)

    # ── a segment ───────────────────────────────────────────────────────────────────────

    async def start(self, instruction: str, max_step_deg: float) -> Plan | str:
        """Open the next epoch and reset the runner for it, then settle the rate, the cap and the
        latency, or say why the segment must not start. The reset queues behind any inference
        the last segment left on the worker, which is how its answer is kept out of this one."""
        self.epoch += 1
        self.instruction = instruction
        reset = await self._ask(self.runner.reset, instruction)
        if isinstance(reset, str):
            return reset
        features = await self._ask(self.runner.features)
        if isinstance(features, str):
            return features
        if not isinstance(features, Features):
            return (
                f"the policy was not started: its features gave {type(features).__name__} "
                "rather than the Features that say its rate, so quackd has no rate to pace it "
                "at. Fix the runner to return one"
            )
        if (refusal := rate_refusal(features)) is not None:
            return refusal
        rate = float(features.rate_hz)
        latency = await self._ask(self.runner.latency_s)
        if isinstance(latency, str):
            return latency
        if not (_finite(latency) and latency >= 0 and math.isfinite(float(latency) * rate)):
            return (
                f"the policy was not started: it declares a latency of {latency!r} s, and a "
                "latency is a number of seconds, 0 or more, that comes to a number of ticks"
            )
        ticks = latency_ticks(float(latency), rate)
        if features.per_tick and ticks > 1:
            return (
                f"the policy was not started: it has to be asked every tick, and it declares "
                f"{float(latency):g} s to answer, which is {ticks} ticks at {rate:g} Hz. quackd "
                "asks a policy again only once it has answered, so one asked every tick has to "
                "answer within a tick: run it where it answers faster"
            )
        return Plan(
            features=features,
            period_s=1.0 / rate,
            cap_deg=speed_cap(max_step_deg, rate),
            latency_ticks=ticks,
        )

    async def run(
        self, arm: LeRobotReal, plan: Plan, segment: Segment, started: float, registers_at: float
    ) -> SegmentEnd:
        """The segment's ticks, from `started` on the arm's clock, until one of them ends it.

        Each tick reads the arm (the registers every `REGISTER_PERIOD_S`) and judges that
        reading before it sends anything: `holding` ends a `pick`, a hot joint, torque off or a
        read the arm did not answer ends any segment with the arm held. Then it takes in what
        the runner has answered, asks it again where the queue is running low, and plays this
        tick's action, which `LeRobotReal._policy_goals` and `clipped_too_long` judge before it
        is sent. What ends it says so in a `SegmentEnd`, with the segment's counts.

        The runner's own error, raised where its answer is judged, ends it `error` with the
        counts up to there, and with no hold: the verb waiting on it stops the arm."""
        run = _Run(started, registers_at, arm._range_clips)
        try:
            return await self._ticks(arm, plan, segment, run)
        except Exception as e:
            arm._policy_error = f"{type(e).__name__}: {e}"
            return self._end(arm, run, "error", f"the policy raised {arm._policy_error}")

    async def _ticks(self, arm: LeRobotReal, plan: Plan, segment: Segment, run: _Run) -> SegmentEnd:
        """The ticks themselves, counted on the segment's `_Run`, which `run` still has to
        report from when one of them raises."""
        started = run.started
        lockstep = bool(getattr(arm.clock, "lockstep", False))
        tick = 0
        while not arm._closed:
            ticked = time.perf_counter()
            now = arm.now()
            if segment.max_s is not None and now - started >= segment.max_s:
                return self._end(arm, run, "time", f"its {segment.max_s:g} s ran out")
            registers = now - run.registers_at >= REGISTER_PERIOD_S
            try:
                obs = await arm._loop_read(registers=registers)
            except Exception as e:
                said = arm.stop_error or f"{type(e).__name__}: {e}"
                return await self._held(arm, run, f"the arm did not answer a read ({said})")
            if registers:
                run.registers_at = now
            if segment.ends_on_holding and arm._holding():
                return self._end(arm, run, "holding", "the gripper closed and settled on something")
            if (why := arm._unsafe_to_drive()) is not None:
                return await self._held(arm, run, why)
            if plan.features.per_tick and not lockstep and self._unanswered():
                # asked last tick and not answered by this one: a tick it never saw
                return await self._missed(
                    arm, run, f"it had not answered tick {tick - 1} by tick {tick}"
                )
            self._take_in(run, tick, plan)
            if self._wants(run, tick, plan, segment):
                if (why := await arm._loop_frames(obs)) is not None:
                    return await self._held(arm, run, why)
                ended = await self._request(arm, run, Observation(tick, obs), plan, lockstep)
                if ended is not None:
                    return ended
            while run.queue and run.queue[0][0] < tick:
                run.queue.popleft()  # meant for a tick the pacer skipped
            action = run.queue.popleft()[1] if run.queue and run.queue[0][0] == tick else None
            if action is None:
                ended = await self._idle(arm, run, segment, now)
                if ended is not None:
                    return ended
            else:
                run.starved_since = None
                ended = await self._play(arm, run, segment, action, now)
                if ended is not None:
                    return ended
            arm._tick_timing.add(time.perf_counter() - ticked)
            run.ticks += 1
            following = await self._pace(arm, run, plan, started, tick + 1)
            if isinstance(following, SegmentEnd):
                return following
            tick = following
        return self._end(arm, run, "guard", "the arm's transport was closed")

    async def _pace(
        self, arm: LeRobotReal, run: _Run, plan: Plan, started: float, following: int
    ) -> int | SegmentEnd:
        """Sleep to tick `following`'s deadline and say which tick it is on waking, or end the
        segment where a runner in tick mode lost one.

        A tick that overran that deadline goes to the next whole period strictly after now,
        and the ticks between are skipped rather than sent late, two to a period. The sleep
        never wakes a tick short of its deadline: the simulator's clock rounds a sleep to its
        nearest whole step, which for a period that is not a whole number of steps can land
        up to half a step early, in the period before the tick's own, and what is left is
        slept again. A clock whose step is longer than a period can also wake past a later
        tick's deadline, and that later tick is the one it is on, the ones between skipped."""
        now = arm.now()
        if now > started + following * plan.period_s:
            after = max(following + 1, math.floor((now - started) / plan.period_s) + 1)
            if (ended := await self._skip(arm, run, plan, after - following)) is not None:
                return ended
            following = after
        deadline = started + following * plan.period_s
        await arm.clock.sleep(deadline - now)
        while (left := deadline - arm.now()) > EXACT:
            before = arm.now()
            await arm.clock.sleep(left)
            if arm.now() <= before:
                break  # a clock a sleep does not move is not slept on again
        woke = math.floor((arm.now() - started) / plan.period_s + EXACT)
        if woke > following:
            if (ended := await self._skip(arm, run, plan, woke - following)) is not None:
                return ended
            following = woke
        return following

    async def _skip(
        self, arm: LeRobotReal, run: _Run, plan: Plan, missed: int
    ) -> SegmentEnd | None:
        """Count `missed` ticks the pacer skipped, which a runner in tick mode never survives."""
        run.skipped += missed
        if plan.features.per_tick:
            return await self._missed(arm, run, f"the pacer skipped {missed} of its ticks")
        return None

    # ── one tick's parts ────────────────────────────────────────────────────────────────

    def _outstanding(self) -> bool:
        """A request of this segment's that has not been taken in yet, answered or not: the one
        request a segment may have out. `_take_in` and `_request` let go of each they take in,
        so an answer that lands between a tick's look and its next ask still counts as out,
        and is taken in on the tick after rather than asked over and lost."""
        req = self._inflight
        return req is not None and req.epoch == self.epoch

    def _unanswered(self) -> bool:
        """A request of this segment's that its runner has not answered yet."""
        req = self._inflight
        return req is not None and req.epoch == self.epoch and not req.future.done()

    def _take_in(self, run: _Run, tick: int, plan: Plan) -> None:
        """Take in whatever has come back by this tick: a request answered on the runner's
        worker, and on a lockstep clock the requests held back until now."""
        req = self._inflight
        if req is not None and req.future.done():
            self._inflight = None
            self._received(run, req, tick, plan)
        while run.held_back and run.held_back[0][0] <= tick:
            _, held = run.held_back.popleft()
            self._received(run, held, tick, plan)

    def _received(self, run: _Run, req: _Request, tick: int, plan: Plan) -> None:
        """One answered request, at the tick it is taken in: thrown away when it is another
        segment's or answers another tick, raised when the runner raised, merged otherwise."""
        if req.future.cancelled():
            return
        if req.epoch != self.epoch:
            run.stale += 1
            return
        chunk = req.future.result()  # the runner's own error ends the segment
        if not isinstance(chunk, Chunk) or chunk.tick != req.tick:
            run.stale += 1
            return
        self._merge(run, chunk, tick, plan)

    def _merge(self, run: _Run, chunk: Chunk, tick: int, plan: Plan) -> None:
        """Put a chunk's actions in the queue at `tick`, the tick it has arrived at."""
        if chunk.done:
            run.done = True
        if not chunk.actions:
            return
        if plan.features.per_tick:
            # asked every tick, it answers for the tick its answer is played at
            run.chunks += 1
            run.got_first = True
            run.queue = deque([(tick, chunk.actions[0])])
            return
        fresh = [
            (chunk.tick + i, action)
            for i, action in enumerate(chunk.actions)
            if chunk.tick + i >= tick
        ]
        run.last_len = len(chunk.actions)
        if not fresh:
            # every action was for a tick already played: counted as thrown away, and not as
            # the segment's first chunk, whose grace still runs for the next
            run.late += 1
            return
        run.chunks += 1
        run.got_first = True
        # the unplayed rest replaces the queue from its first tick on, never appended to it
        kept = deque(entry for entry in run.queue if entry[0] < fresh[0][0])
        kept.extend(fresh)
        run.queue = kept

    def _wants(self, run: _Run, tick: int, plan: Plan, segment: Segment) -> bool:
        """Whether this tick asks the runner for more: never after it said it was done or once
        the segment has played its chunks, never with a request of its own outstanding, every
        tick in tick mode, and otherwise once the queue is down to `REFILL_SHARE` of the last
        chunk. A chunk is one that had something to play (`_merge`), so an answer thrown away
        is asked over rather than counted, and a runner whose answers never play ends on the
        grace or `STARVE_S` rather than on its chunks."""
        if run.done or self._outstanding():
            return False
        if segment.max_chunks is not None and run.chunks >= segment.max_chunks:
            return False
        if plan.features.per_tick:
            return True
        if run.held_back:
            return False  # a request held back on its way is the one outstanding
        left = sum(1 for at, _ in run.queue if at >= tick)
        return left == 0 or left <= REFILL_SHARE * run.last_len

    async def _request(
        self, arm: LeRobotReal, run: _Run, observation: Observation, plan: Plan, lockstep: bool
    ) -> SegmentEnd | None:
        """Ask the runner for a chunk, stamped with this tick. On a lockstep clock the answer is
        awaited here, with time standing still, and held back its `k` ticks whole, to be judged
        when it is let go as the arm would judge it on arrival. A runner that answers at once is
        awaited here on the wall's clock too, within what is left of the segment's patience.
        Otherwise the answer is left on the worker and taken in on the tick it has come back
        by."""
        future = self._submit(self.runner.next_chunk, observation, dict(run.last_sent))
        req = _Request(self.epoch, observation.tick, future)
        self._inflight = req
        if not lockstep and plan.latency_ticks > 0:
            return None
        patience = None
        if not lockstep:
            patience = (
                self.starve_s
                if run.got_first
                else max(0.0, self.first_chunk_s - (arm.now() - run.started))
            )
        if not await self._answered(future, patience):
            return await self._starved(arm, run)
        self._inflight = None
        if lockstep and plan.latency_ticks > 0:
            run.held_back.append((req.tick + plan.latency_ticks, req))
            return None
        self._received(run, req, observation.tick, plan)
        return None

    async def _idle(
        self, arm: LeRobotReal, run: _Run, segment: Segment, now: float
    ) -> SegmentEnd | None:
        """A tick with nothing to send: a runner that is done, a segment whose chunks are all
        played, or starvation, which the grace and `STARVE_S` bound."""
        run.still = None  # an arm left alone is not a stalled one
        outstanding = self._outstanding() or bool(run.held_back)
        if run.done:
            if segment.ends_on_holding:
                # the grasp may still be closing: read on, send nothing, and look again
                if run.done_at is None:
                    run.done_at = now
                elif now - run.done_at >= PICK_SETTLE_S:
                    return self._end(
                        arm,
                        run,
                        "finished",
                        "the policy finished and the gripper did not settle on anything",
                    )
                return None
            return self._end(
                arm, run, "chunks", f"the policy said it was done after {run.chunks} chunks"
            )
        if segment.max_chunks is not None and run.chunks >= segment.max_chunks and not outstanding:
            return self._end(arm, run, "chunks", f"its {run.chunks} chunks were played")
        run.starved += 1
        if run.starved_since is None:
            run.starved_since = now
        since = run.starved_since if run.got_first else run.started
        if now - since >= self._patience(run):
            return await self._starved(arm, run)
        return None

    async def _play(
        self,
        arm: LeRobotReal,
        run: _Run,
        segment: Segment,
        action: Mapping[str, Any],
        now: float,
    ) -> SegmentEnd | None:
        """Judge this tick's action and send it, or end the segment on it."""
        goals, why = arm._policy_goals(action)
        if why is None:
            why = clipped_too_long(goals, run.clipped_since, now, arm.joint_range_deg)
        if why is not None:
            return await self._held(arm, run, why)
        try:
            sent = await arm._send(goals)
        except Exception as e:
            run.failed_sends += 1
            if run.failed_sends >= FAILED_SENDS:
                said = arm.stop_error or f"{type(e).__name__}: {e}"
                return await self._held(
                    arm,
                    run,
                    f"{run.failed_sends} sends in a row did not reach the arm, the last with "
                    f"{said}",
                )
            return None
        run.failed_sends = 0
        if sent:
            # what went out, after the clip and LeRobot's step cap, for the runner's next ask
            run.last_sent = dict(sent)
        if segment.ends_on_stall:
            return await self._stall(arm, run, now)
        return None

    async def _stall(self, arm: LeRobotReal, run: _Run, now: float) -> SegmentEnd | None:
        """End `manipulate` once every joint has read within `STALL_DEG` of where it was for
        `STALL_S`, on ticks that sent, with the arm held where it is, as a verb's own stall
        holds it (`verbs._drive`). An arm stopped by something in its way still has the
        policy's last goal past it, and the servo would push on toward it through the pilot's
        thinking. The hold leaves the gripper's goal alone, so a grasp keeps its squeeze."""
        joints = {j: arm._joints[j] for j in JOINTS if j in arm._joints}
        anchor = run.still
        if anchor is None or any(
            abs(joints[j] - anchor[1].get(j, joints[j])) > STALL_DEG for j in joints
        ):
            run.still = (now, joints)
            return None
        if now - anchor[0] >= STALL_S:
            return await self._held(
                arm,
                run,
                f"the arm moved less than {STALL_DEG:g} degrees in {STALL_S:g} s of goals",
                how="stall",
            )
        return None

    # ── endings ─────────────────────────────────────────────────────────────────────────

    def _patience(self, run: _Run) -> float:
        """How long the segment may go without an action: `FIRST_CHUNK_S` until its first chunk,
        and `STARVE_S` after it."""
        return self.starve_s if run.got_first else self.first_chunk_s

    async def _starved(self, arm: LeRobotReal, run: _Run) -> SegmentEnd:
        """End a segment that went its whole patience without an action, which is said as that
        bound, however little of it the last wait had left."""
        which = "a first chunk" if not run.got_first else "a chunk"
        return await self._held(
            arm,
            run,
            f"the policy gave no action for {self._patience(run):g} s, waiting for {which}",
            how="starved",
        )

    async def _missed(self, arm: LeRobotReal, run: _Run, why: str) -> SegmentEnd:
        """A runner asked every tick has missed one: hold the arm, reset it on its worker behind
        whatever it is still doing, and end."""
        ended = await self._held(
            arm, run, f"the policy has to be asked every tick, and {why}", how="guard"
        )
        self._submit(self.runner.reset, self.instruction)
        return ended

    async def _held(
        self, arm: LeRobotReal, run: _Run, why: str, *, how: SegmentHow = "guard"
    ) -> SegmentEnd:
        return self._counted(arm, run, await arm._held(why, how=how))

    def _end(self, arm: LeRobotReal, run: _Run, how: SegmentHow, reason: str) -> SegmentEnd:
        return self._counted(arm, run, SegmentEnd(how, reason))

    @staticmethod
    def _counted(arm: LeRobotReal, run: _Run, ended: SegmentEnd) -> SegmentEnd:
        elapsed = arm.now() - run.started
        stats = SegmentStats(
            ticks=run.ticks,
            skipped=run.skipped,
            chunks=run.chunks,
            starved=run.starved,
            stale=run.stale + run.late,
            clips=arm._range_clips - run.clips_at,
            hz=round(run.ticks / elapsed, 1) if elapsed > 0 else None,
        )
        return dataclasses.replace(ended, stats=stats)


__all__ = [
    "CLIP_SUSTAIN_S",
    "FAILED_SENDS",
    "FIRST_CHUNK_S",
    "MAX_RATE_HZ",
    "MIN_RATE_HZ",
    "REGISTER_PERIOD_S",
    "RESET_S",
    "STALL_S",
    "STARVE_S",
    "Plan",
    "PolicyLoop",
    "Segment",
    "clipped_too_long",
    "latency_ticks",
    "rate_refusal",
    "speed_cap",
]
