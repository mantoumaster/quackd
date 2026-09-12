# PLAN.md — quackd

What is still open. Everything that has shipped is in [CHANGELOG.md](CHANGELOG.md), the
[ADRs](docs/adr/) and the git history, which record it better than a task list can.

Legend: 🔨 in progress · ⬜ todo · ⏸ blocked (with reason)

## Only a human can

Six bring-ups, one per body. Each needs hardware quackd has never touched, and each ends the
same way: flip that backend's row in [`docs/adapter-status.md`](docs/adapter-status.md), and
not before.

- ⏸ **An Open Duck Mini v2**, the most reachable of the six because you can build it. Run
  `open_duck:bridge` against a duck you built, work through
  [docs/open-duck-hardware-checklist.md](docs/open-duck-hardware-checklist.md), and confirm
  the deadman by pulling Wi-Fi mid-walk. Then the five numbers at the end of that checklist:
  boot time against the watchdog budget, camd's peak memory against its cap, the observed
  loop-rate floor, the camera's field of view against a tape measure, and the accelerometer
  upright versus on its side. The last one is what would give this robot fall detection, and
  quackd deliberately does not guess it, because a wrong fall detector fails as a confident
  "not fallen".
- ⏸ **A Microduck.** Run `--robot microduck:jsonrpc` against a real `robotd` and work through
  [docs/microduck-hardware-checklist.md](docs/microduck-hardware-checklist.md), whose step 0
  now rehearses the whole pilot in the physics simulator first. The path is built and audited:
  pinned at a commit and bumped to API v23 (it was v16 against a moving link, so the handshake
  would have refused), state actually subscribed to, and video over `webrtc://` because
  upstream serves no frames on the socket. Pre-orders opened 2026-08-27, earliest arrivals
  estimated around Christmas 2026 and later orders four to six months out.
- ⏸ **A ToddlerBot** on its safety stand, running `bridge/toddlerbot/quackd_toddlerbot_bridge.py`
  through [docs/toddlerbot-hardware-checklist.md](docs/toddlerbot-hardware-checklist.md). What
  most needs a real robot: whether the safe-pose slew is safe from a crawl, what tilt really
  means fallen, whether the neck axes are what the motor names imply, and whether a calibrated
  zero survives a restart.
- ⏸ **An XLeRobot.** Start the host (it is commented out of upstream's own package `__init__`
  and exits after an hour) and point `xlerobot-lookout` at it. What most needs a real cart: the
  camera colour order, whether `+x` is really forward, and whether the head motors are what
  upstream's agent library implies.
- ⏸ **An AlohaMini.** Start `bridge/alohamini/quackd_alohamini_host.py` rather than upstream's
  host and point `alohamini-lookout` at it. What most needs a real robot: whether `+x` is
  physically forward, the camera colour order, how fast the lift travels in mm/s, and whether
  the wrapper really does leave the arms holding.
- ⏸ **An SO-101 arm or any rosbridge base.** `lerobot:real` against a calibrated arm,
  `rosbridge:ws` against a bridge. A flock across two machines needs a distributed clock first.

## Open here

- 🔨 **Somebody has to drive `web/` in a browser, and record it.** The page itself is no longer
  unopened: it booted clean twice on the machine that wrote it (`8d72a2a`, `a9fea18`), with the
  fonts and the mark loaded and a held `W` walking the duck. What that leaves is everything past
  the boot — a full model-driven run, a key barging in out of one, the Record button, the switch
  thrown mid-run, and any browser or machine but that one. None of it was recorded, so there is
  no asset and nothing anybody else can check. Separately, the four measured claims in
  `web/README.md` come from a scratch harness that is not in the repository, and both files it
  measured have changed since, in the abort path and in the arena's geometry, so nothing here can
  re-run it. Locally it is `python web/serve.py`, then <http://localhost:8000/simulator/>.
- ⬜ Flock mode does not know `open_duck` yet (`flock/runner.py` knows one adapter), and a
  hardware flock waits on Microducks shipping.
- ⏸ **A real model recording**, in either simulator, to replace a scripted-pilot asset and drop
  the label (see [docs/assets](docs/assets/README.md)). Needs a key.
- ⬜ **The browser demo is not at parity with the backend.** Seven of the manifest's fifteen
  verbs and none of the three composites, a contract of its own, an arena that is not upstream's
  scene, geometric perception, no hash check on anything it fetches, a seed that means the same
  distributions and not the same layout, and no scripted pilot. `web/README.md` holds that list
  in full and is the one place it is kept; this bullet is the reminder that it is a list of open
  gaps and not just a disclosure.
- ⬜ **`GAIT_FLOOR_VY` was never measured.** The forward and turning floors were; the sideways
  one is assumed equal to the training maximum, so every lateral request is sent at full
  scale. The assumption is in `GAIT_THRESHOLD`'s note and in the state's `assumptions`, and
  the fix is the same script that produced the other two.
- 🔨 **A transcript from a live local server** (Ollama, vLLM, llama.cpp). None on the dev
  machine. PR #5's contributor reports `find-and-kick` against Qwen 2.5 Coder 14B through LM
  Studio on seeds 5 and 6, both successes with memory read and written, but no transcript from
  it is in the repository, so the README says exactly that.
- ⏸ **Exercise `remember` against a cloud model.** The scripted pilot has no script for it, so
  `--provider fake` writes episodes and never a note.
- ⏸ Verify the `gpt-5`, `grok-4` and `gemini-2.5-pro` default IDs against vendor docs. All are
  overridable with `QUACKD_MODEL`.
- ⏸ Upload `docs/assets/social-preview.png` under Settings → Social preview. There is no API
  for it.

## Open elsewhere

One item, and it closed: the half of the browser demo that no commit here could finish.

- ✅ **`web/` is on the web.** <https://www.quackd.org/simulator> answers. That address belongs to
  quackd-web, a separate Vercel project serving the landing page, and its build now fetches this
  directory into its own `/simulator` at a pinned commit — so a change here reaches the page on
  that project's next deploy, and `/simulator/source.json` records which commit the live copy came
  from. The landing page points at the demo from five places; the demo's header points back.

## Release checklist

The one reusable thing the shipped milestones left behind. Every release since 0.1.0 has run
this, and the per-release detail is in [CHANGELOG.md](CHANGELOG.md).

1. All four CI gates green on `main`: `ruff check`, `ruff format --check`, `mypy` on 3.11 and
   3.12, `pytest` with `QUACKD_STRICT_SEEDS=1`, plus `quackd validate ducks/*.duck`.
2. Read the release note against the code before it ships. Every release so far has found
   claims that had gone stale between writing and tagging.
3. Annotated tag, pushed with `main`.
4. GitHub Release on `main` with the wheel and the sdist attached.
5. Publish to PyPI, and check the SHA256 of both files is identical in both places.
6. `uvx --from quackd==<version> quackd run find-and-kick --provider fake` from a clean
   install, twice, so the second run reads the first one's episode.
7. Update the About description (GitHub's cap is 350 characters) and Topics (cap 20).
