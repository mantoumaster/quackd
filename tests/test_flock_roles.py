"""Heterogeneous flock primitives: the capability term, the role auction, the new messages."""

from __future__ import annotations

import random
from types import SimpleNamespace as NS

from pydantic import TypeAdapter

from quackd.duckfile.schema import FlockRole
from quackd.flock.auction import AuctionPolicy, RoleAuction
from quackd.flock.capability import eligible_roles, missing
from quackd.flock.messages import (
    BidMsg,
    FlockMessage,
    Hint,
    HintMsg,
    ResultMsg,
    RoleMsg,
    VerdictMsg,
)

ROLES = {
    "spotter": FlockRole(requires=["observe", "gaze"]),
    "kicker": FlockRole(requires=["go_to", "kick"]),
}
DUCK = ["observe", "report_state", "stop", "say", "move", "go_to", "search_scan", "kick", "gaze"]
HEAD = ["observe", "report_state", "stop", "say", "search_scan", "gaze", "express"]


def test_capability_term_is_alias_aware() -> None:
    assert missing(["get_frame", "walk_to"], DUCK) == []
    assert missing(["go_to", "kick"], HEAD) == ["go_to", "kick"]
    assert eligible_roles(ROLES, DUCK) == ["kicker", "spotter"]  # a duck can spot too
    assert eligible_roles(ROLES, HEAD) == ["spotter"]
    assert eligible_roles(ROLES, ["stop"]) == []


def _bid(src: str, role: str, dist: float, t: float = 0.0) -> BidMsg:
    return BidMsg(t=t, src=src, task_id="t", ball_dist_m=dist, role=role, provides=[])


def test_role_auction_fills_most_constrained_role_first() -> None:
    now = NS(t=0.0)
    auction = RoleAuction(AuctionPolicy(window_s=0.4), lambda: now.t, ROLES)
    auction.open(_bid("spotter-01", "spotter", 1.1))
    assert not auction.complete(set())  # nobody bid for kicker yet
    auction.add(_bid("duck-01", "spotter", 0.78))  # the duck is closer, and could spot...
    auction.add(_bid("duck-01", "kicker", 0.78))  # ...but it is the only one who can kick
    assert auction.complete(set())
    now.t = 0.5
    assert auction.due()
    decision = auction.decide({}, set())
    assert decision is not None
    assert decision.assignments == {"kicker": "duck-01", "spotter": "spotter-01"}
    assert decision.kicker == "duck-01" and decision.costs["spotter"] == 1.1
    assert decision.ties == [] and decision.hysteresis_applied == []


def test_role_auction_is_independent_of_bid_order() -> None:
    bids = [
        _bid("duck-01", "kicker", 0.9),
        _bid("duck-02", "kicker", 0.7),
        _bid("duck-01", "spotter", 0.9),
        _bid("duck-02", "spotter", 0.7),
        _bid("spotter-01", "spotter", 1.3),
    ]
    outcomes = set()
    for seed in range(20):
        shuffled = list(bids)
        random.Random(seed).shuffle(shuffled)
        auction = RoleAuction(AuctionPolicy(), lambda: 0.0, ROLES)
        auction.open(shuffled[0])
        for b in shuffled[1:]:
            auction.add(b)
        decision = auction.decide({}, set())
        assert decision is not None
        outcomes.add(tuple(sorted(decision.assignments.items())))
    assert outcomes == {(("kicker", "duck-02"), ("spotter", "duck-01"))}


def test_role_auction_holds_the_spotter_and_applies_hysteresis_per_role() -> None:
    auction = RoleAuction(AuctionPolicy(hysteresis=0.2), lambda: 0.0, ROLES)
    auction.held = {"spotter": "spotter-01"}
    auction.open(_bid("duck-01", "kicker", 1.0))  # the previous kicker
    auction.add(_bid("duck-02", "kicker", 0.85))  # 15 % better: not enough
    auction.add(_bid("spotter-01", "spotter", 0.5))  # ignored: the role is held
    decision = auction.decide({"kicker": "duck-01"}, set())
    assert decision is not None
    assert decision.assignments == {"spotter": "spotter-01", "kicker": "duck-01"}
    assert decision.hysteresis_applied == ["kicker"]
    auction.open(_bid("duck-02", "kicker", 0.7))  # 30 % better: unseats
    auction.add(_bid("duck-01", "kicker", 1.0))
    decision = auction.decide({"kicker": "duck-01"}, set())
    assert decision is not None and decision.assignments["kicker"] == "duck-02"


def test_role_auction_is_void_when_a_role_cannot_be_filled() -> None:
    auction = RoleAuction(AuctionPolicy(), lambda: 0.0, ROLES)
    auction.open(_bid("duck-01", "kicker", 0.5))
    auction.add(_bid("duck-01", "spotter", 0.5))  # one duck cannot take both roles
    assert not auction.complete(set())
    assert auction.decide({}, set()) is None
    auction.open(_bid("duck-01", "kicker", 0.5))
    auction.add(_bid("spotter-01", "spotter", 1.0))
    assert auction.decide({}, {"spotter-01"}) is None  # the only spotter is excluded
    auction.open(_bid("duck-01", "dancer", 0.5))  # an unknown role is dropped on add
    assert auction.bids == {}


def test_new_messages_round_trip_and_old_ones_keep_their_defaults() -> None:
    adapter: TypeAdapter[FlockMessage] = TypeAdapter(FlockMessage)
    hint = Hint(target="ball", x_m=0.2, y_m=-0.4, by="spotter-01", est_dist_m=0.9, bearing_deg=12)
    for msg in (
        HintMsg(t=1.0, src="spotter-01", task_id="t", hint=hint),
        VerdictMsg(
            t=2.0,
            src="spotter-01",
            task_id="t",
            target="ball",
            kicker="duck-01",
            verdict="moved",
            moved_m=0.61,
            ref={"x_m": 0.2, "y_m": -0.4},
            seen={"x_m": 0.8, "y_m": -0.3},
            frames=2,
        ),
        RoleMsg(
            t=3.0,
            src="coordinator",
            task_id="t",
            duck="spotter-01",
            role="JUDGE",
            kicker="duck-01",
            seq=5,
        ),
        ResultMsg(t=4.0, src="duck-01", task_id="t", status="kick_done", ball_moved_m=0.6),
    ):
        again = adapter.validate_python(msg.model_dump())
        assert again == msg and again.kind == msg.kind
    legacy = adapter.validate_python(
        {"kind": "BID", "t": 0.0, "src": "duck-1", "task_id": "t", "ball_dist_m": 0.6}
    )
    assert isinstance(legacy, BidMsg) and legacy.role is None and legacy.provides == []
    role = adapter.validate_python(
        {
            "kind": "ROLE",
            "t": 0.0,
            "src": "coordinator",
            "task_id": "t",
            "duck": "duck-1",
            "role": "KICK",
        }
    )
    assert isinstance(role, RoleMsg) and role.seq == 0 and role.flock_role is None
