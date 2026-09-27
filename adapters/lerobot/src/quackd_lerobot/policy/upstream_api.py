"""The only file in quackd allowed to spell a LeRobot policy name (ADR-0022).

Every constant is tagged VERIFIED (read from upstream source at the pin, link given) or
UNVERIFIED (an assumption of ours, with what quackd does about it). `docs/adapters/lerobot.md`
is the human-readable version; `tests/test_upstream_api.py` proves UNVERIFIED names are only
reachable from the files that live with them.

These are read against lerobot 0.6.1, the version the laptop that drives the lab arm runs, at
the commit its `v0.6.1` tag names (7e241bd630a3719a56157a497ce5d08f244784f1), and not at the
`main` commit the arm's own refs are pinned to (`quackd_lerobot.upstream_api.PIN`). A policy
runs where a checkpoint was trained and exported, and its processors and its config are read
by whatever lerobot the policy server has installed, so the version people run is the one to
read. The installed 0.6.1 wheel and the tag were compared file by file on the day they were
read, and every file these rows cite was the same.

The policy refs used to live in the arm's own file. They moved here when quackd grew a policy
server (`server.py`), because the server is where a checkpoint is loaded, and the arm's process
never imports torch or LeRobot's policies at all: `client.py` reaches the server over HTTP.
Nothing in this file has been exercised against a checkpoint yet: the server serves scripted
policies only (`scripted.py`), and POLICY_PIPELINE says what that leaves unproven.
"""

from __future__ import annotations

from quackd.upstream import UpstreamRef

REPO = "https://github.com/huggingface/lerobot"
PIN = "7e241bd630a3719a56157a497ce5d08f244784f1"  # the v0.6.1 tag
VERSION = "0.6.1"
READ_ON = "2026-09-27"


def src(path: str, line: int | None = None) -> str:
    return f"{REPO}/blob/{PIN}/{path}" + (f"#L{line}" if line else "")


_POLICY = "src/lerobot/policies/pretrained.py"
_FACTORY = "src/lerobot/policies/factory.py"
_POLICY_CFG = "src/lerobot/configs/policies.py"
_UTILS = "src/lerobot/policies/utils.py"
_ACT_CFG = "src/lerobot/policies/act/configuration_act.py"
_SMOLVLA_CFG = "src/lerobot/policies/smolvla/configuration_smolvla.py"
_PIPELINE = "src/lerobot/processor/pipeline.py"
_TOKENIZER = "src/lerobot/processor/tokenizer_processor.py"
_ASYNC = "src/lerobot/async_inference"

# ── a policy (moved here from the arm's own file, and read again at 0.6.1) ──────────────

POLICY_BASE = UpstreamRef(
    "lerobot.policies.pretrained.PreTrainedPolicy", "VERIFIED", src(_POLICY, 105)
)
PRETRAINED_CONFIG = UpstreamRef(
    "lerobot.configs.policies.PreTrainedConfig",
    "VERIFIED",
    src(_POLICY_CFG, 41),
    "`class PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC)`, with "
    "`from_pretrained(pretrained_name_or_path, *, ...)` (line 172), a `device` field (line 62) "
    "and a `type` property (line 99). A checkpoint's config is read through it before the "
    "policy class is built, so it is the one policy name that is not in the factory",
)
CONFIG_HAS_NO_RATE = UpstreamRef(
    "a policy's config carries no fps",
    "VERIFIED",
    src(_POLICY_CFG, 41),
    "nothing in PreTrainedConfig or the policy configs that extend it says how many actions a "
    "second the policy was trained to run at: that is the fps of the dataset it learned from. "
    "So the policy server takes a rate it is given or reads one from where it names, and never "
    "guesses one, and the rate travels to the arm with the source it came from",
)
POLICY_FROM_PRETRAINED = UpstreamRef(
    "PreTrainedPolicy.from_pretrained(path, *, config=None, local_files_only=False, "
    "revision=None, strict=False)",
    "VERIFIED",
    src(_POLICY, 169),
    "a local directory or a Hub repo id, at a revision when one is given; the policy comes back "
    "in eval mode (line 226)",
)
POLICY_SELECT_ACTION = UpstreamRef(
    "PreTrainedPolicy.select_action(batch: dict[str, Tensor]) -> Tensor",
    "VERIFIED",
    src(_POLICY, 280),
    "one action per call, the policy handles its own action-chunk cache",
)
POLICY_PREDICT_ACTION_CHUNK = UpstreamRef(
    "PreTrainedPolicy.predict_action_chunk(batch: dict[str, Tensor]) -> Tensor",
    "VERIFIED",
    src(_POLICY, 271),
    "the whole chunk for one observation, which is what the policy server answers a step with, "
    "one action per tick from the tick the observation was read at",
)
POLICY_RESET = UpstreamRef("PreTrainedPolicy.reset()", "VERIFIED", src(_POLICY, 245))
GET_POLICY_CLASS = UpstreamRef(
    "lerobot.policies.factory.get_policy_class(name)", "VERIFIED", src(_FACTORY, 79)
)
MAKE_PRE_POST_PROCESSORS = UpstreamRef(
    "lerobot.policies.factory.make_pre_post_processors(policy_cfg, pretrained_path=None, "
    "pretrained_revision=None)",
    "VERIFIED",
    src(_FACTORY, 150),
    "a raw observation goes through the pre-processor and the action tensor through the "
    "post-processor before it is a RobotAction, and both load from the checkpoint at the "
    "revision given",
)
MAKE_POLICY = UpstreamRef(
    "lerobot.policies.factory.make_policy(cfg)", "VERIFIED", src(_FACTORY, 240)
)
BUILD_INFERENCE_FRAME = UpstreamRef(
    "lerobot.policies.utils.build_inference_frame(observation, device, ds_features, task, "
    "robot_type)",
    "VERIFIED",
    src(_UTILS, 141),
    "picks the keys `ds_features` names out of a raw observation and makes them tensors on the "
    "device, which is why the server builds `ds_features` from the motor names and the cameras "
    "a reset declares rather than from the checkpoint alone",
)
MAKE_ROBOT_ACTION = UpstreamRef(
    "lerobot.policies.utils.make_robot_action(action_tensor, ds_features)",
    "VERIFIED",
    src(_UTILS, 175),
    "one action row to a dict named by `ds_features`' action names, with a batch dimension "
    "squeezed off, so a chunk is turned into actions a row at a time",
)

# ── chunks ──────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE = UpstreamRef(
    "ACTConfig.chunk_size and n_action_steps",
    "VERIFIED",
    src(_ACT_CFG, 85),
    "`chunk_size` is how many actions one inference predicts and `n_action_steps` how many of "
    "them are played before the next (line 86), both counted in environment steps, and "
    "`n_action_steps` may not exceed `chunk_size` (line 143). SmolVLA's defaults are 50 and 50 "
    f"({src(_SMOLVLA_CFG, 29)}). The policy server reports both",
)
TEMPORAL_ENSEMBLE = UpstreamRef(
    "ACTConfig.temporal_ensemble_coeff",
    "VERIFIED",
    src(_ACT_CFG, 119),
    "None by default. Set, ACT is asked every step and `n_action_steps` must be 1 (line 138), "
    "which is a policy asked every tick: the loop's tick mode",
)

# ── a checkpoint is code (why a policy never runs beside the arm's bus) ─────────────────

PROCESSOR_CLASS_IMPORT = UpstreamRef(
    "a processor step named by class is imported by its module path",
    "VERIFIED",
    src(_PIPELINE, 1085),
    "PolicyProcessorPipeline.from_pretrained resolves each step of a processor's JSON by its "
    "`registry_name`, or else imports whatever `module.Class` its `class` key names with "
    "importlib, so loading a checkpoint's processors can run any code the checkpoint points "
    "at. This is why no checkpoint is loaded in the process that owns the serial bus",
)
TOKENIZER_TRUSTS_REMOTE_CODE = UpstreamRef(
    "TokenizerProcessorStep.trust_remote_code defaults to True",
    "VERIFIED",
    src(_TOKENIZER, 348),
    "a tokenizer step loads its tokenizer trusting the repository's own code unless told not "
    "to, and SmolVLA names its backbone by an unpinned Hub name "
    f"(`vlm_model_name`, {src(_SMOLVLA_CFG, 84)})",
)

# ── LeRobot's own policy server, and why quackd has one of its own ──────────────────────

ASYNC_PICKLE = UpstreamRef(
    "async inference unpickles what it is sent",
    "VERIFIED",
    src(f"{_ASYNC}/policy_server.py", 183),
    "the policy server `pickle.loads` every observation (and the policy spec, line 125), and "
    "the robot client unpickles every chunk it gets back (robot_client.py line 286), over "
    "`add_insecure_port` (line 428), so anything that can reach the port runs code on the "
    "other side. quackd's protocol is JSON with every number checked, behind a token",
)
ASYNC_CLIENT_NEEDS_TORCH = UpstreamRef(
    "the async robot client imports torch",
    "VERIFIED",
    src(f"{_ASYNC}/robot_client.py", 48),
    "and the process that owns the serial bus is the one quackd keeps free of torch",
)
ASYNC_SKIPS_SIMILAR = UpstreamRef(
    "observations_similar(obs1, obs2, lerobot_features, atol=1)",
    "VERIFIED",
    src(f"{_ASYNC}/helpers.py", 281),
    "the async server skips an observation whose joint state is within a norm of 1 of the "
    "last one it ran, which is how a policy stops seeing an arm that is barely moving. quackd's "
    "server runs every step it is asked",
)
ASYNC_SUPPORTED_POLICIES = UpstreamRef(
    'SUPPORTED_POLICIES = ["act", "smolvla", "diffusion", "tdmpc", "vqbet", "pi0", "pi05", '
    '"groot"]',
    "VERIFIED",
    src(f"{_ASYNC}/constants.py", 26),
    "the policies the async server will load, and a policy type outside it is refused there",
)

# ── UNVERIFIED: our assumptions, and what quackd does about each ────────────────────────

POLICY_PIPELINE = UpstreamRef(
    "POLICY_PIPELINE",
    "UNVERIFIED",
    src(_FACTORY, 150),
    "wiring a PreTrainedPolicy end to end (pre-processor, predict_action_chunk, "
    "post-processor, device) has never been run by us. The policy server serves scripted "
    "policies only, and a checkpoint named to it is refused. The arm's backend still offers "
    "load_policy(), which builds a policy object from the verified names and is untested. A "
    "policy's actions go through the same step cap as a verb's, and a goal outside the travel "
    "is clipped and counted, unlike a verb's goal, which is refused (ADR-0036). Both are "
    "quackd's rules and not upstream's",
)


def all_refs() -> list[UpstreamRef]:
    return [v for v in globals().values() if isinstance(v, UpstreamRef)]


def refs_by_status(status: str) -> list[UpstreamRef]:
    return [r for r in all_refs() if r.status == status]
