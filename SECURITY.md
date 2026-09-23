# Security Policy

## Reporting

Please **do not** open a public issue for vulnerabilities. Email
**ksjeno@gmail.com** with "quackd security" in the subject, or use GitHub's private
vulnerability reporting on the repository if enabled. You will get an acknowledgement
within 72 hours.

## What "security" means when an LLM commands a robot

quackd sends *intents* to a robot. How much of the stopping the robot itself does depends
on the body, and each adapter declares it in its manifest's `safety_authority`
(see `docs/safety.md`). On a Microduck, `robotd` is the safety authority: it clamps
velocities, detects falls, and zeroes motion when commands stall. On an Open Duck Mini the
deadman is quackd's own daemon, running on the robot and zeroing the velocity inside the
50 Hz loop, so it is code we ship and therefore code we are answerable for. On the other
five bodies upstream has no deadman that covers the whole body. A rosbridge base
declares `native: none`, and a LeRobot arm has a torque limit on its gripper alone and holds
its last goal.
An XLeRobot's host watchdog zeroes its wheels and leaves the arms holding, and an
AlohaMini's covers the base and the lift and never the arms, so both are partial by
construction. A ToddlerBot has no watchdog, timeout or e-stop anywhere upstream at all, and
cannot get up if it falls, so its only deadman is the one quackd's own daemon runs. On all
five, quackd's own heartbeat and `stop` are most or all of what stops them. That makes the client-side
layer (verb allowlists, confirm gates, budgets, the heartbeat, the kill switch)
security-relevant, not just a convenience. A bug that lets an LLM or an MCP client bypass
it is a security issue.

quackd also ships code that runs **on a robot**, which is a different kind of
surface from everything above. Everything under `bridge/` is in scope in its own right:
two daemons for an Open Duck Mini v2's Raspberry Pi, a host wrapper for an AlohaMini, and
a daemon that walks a ToddlerBot.

Also in scope:

- API keys leaking into transcripts, GIFs, logs, run directories, or a robot's memory file.
  `TYPESAFE_API_KEY`, which the optional discrete stepper reads for the `jev` preset
  ([docs/decision-llms/jev.md](docs/decision-llms/jev.md)), is one of these. It is the only
  decision LLM quackd names that wants a key at all: every other one is a server you run
  yourself or a checkpoint in this process, and quackd hands those a placeholder
  in the key field rather than whatever hosted key happens to be sitting in the same `.env`.
- **The command line, which the run record now holds.** A solo run writes down what it was
  started with, in three places: `command` in the transcript's `run_start`, `command` in
  `summary.json`, and the first line of `terminal.txt`. A flock root has no `run_start`, so
  both flock runners write `command` and `version` into the root `summary.json` themselves,
  and the root `terminal.txt` opens with the same line; a pilot flock's members each keep a
  `run_start` of their own besides. The values of `--api-key` and `--token` are replaced with
  `***` everywhere that line is written, so a reader sees that a key was passed and never
  what it was. The four flags that take a URL, `--base-url`, `--address`, `--camera-url` and
  `--decision-url`, keep the half a reader needs and lose the half that has to be rotated:
  the scheme, the host, the port, the path and the username stay, a password in the URL
  becomes `***`, and so does any query parameter named like a credential (`api_key`, `token`,
  `sig` and the rest of `SECRET_QUERY_KEYS`). `--extra-body`, and `QUACKD_EXTRA_BODY` behind
  it, is a JSON object a vendor asked for and quackd never reads, which makes it exactly
  where an `authorization` header ends up; it reaches the transcript's `run_start` as
  `extra_body`, and it is walked to the bottom on the way in with every credential-named key
  replaced. All of that is redaction **by name**, of flag names and of key names and nothing
  cleverer, which is worth stating plainly because it decides what is safe to paste into an
  issue. A secret typed as the value of some **other** flag is written down in full, and so is
  a credential a vendor asked for under a name these lists do not carry. A key handed to the
  provider through its own environment variable, which is the normal way and the right one, is
  in no part of the record, and `QUACKD_EXTRA_BODY` is the one environment variable that
  reaches it at all. What would be a security issue: the value of either secret flag reaching
  any of the places above, a password or a named credential surviving a URL flag, a
  credential-named key surviving `extra_body`, or a new flag that takes a secret and is in
  neither `SECRET_FLAGS` nor `URL_FLAGS` (`quackd/command.py`).
- **What the discrete stepper is sent** (`quackd run --decision-llm`, off unless you name one,
  [docs/decision-llms.md](docs/decision-llms.md)). Whichever one answers is sent the same
  thing, once a turn: the task's goal, the robot's own description of itself and its last few
  results. Where that goes is the part that differs, and it is worth knowing which of the three
  you chose. `jev` is a third party's hosted API, so a run that names it sends that state over
  the network to TypeSafe ([docs/decision-llms/jev.md](docs/decision-llms/jev.md)). Every
  other server row, `local` included, is a server you run, so it goes wherever
  `--decision-url` points, which is a port on your own machine unless you moved it. What each
  one binds and whether anything authenticates it is on its own page, and the two are not the
  same answer: [kev](docs/decision-llms/kev.md) binds `127.0.0.1` and authenticates nothing,
  while [von](docs/decision-llms/von.md) binds every interface unless you pass `--host`, which
  is why the catalogue's own command passes it. `laya` runs inside this process, so nothing
  leaves it at all ([docs/decision-llms/laya.md](docs/decision-llms/laya.md)). None of them is
  ever sent a camera frame, a system prompt or an API key. How much was sent is on the
  record, as `state_chars` and `state_tokens_est` on each `decision` event, and which fields
  were dropped to fit is there as `trimmed`; the text itself is not, so a reader auditing what
  left the machine is reading a size and a shape rather than the words. The address is on the record too, in `run_start.decision_llm`, with a password in it
  or a credential-named query parameter already replaced by `***`, because that url can arrive
  through `QUACKD_DECISION_URL` where argv redaction would never see it. What would be a
  security issue: a picture or a credential reaching any of them, a credential surviving that
  recorded url, a hosted key being sent to a server you run, or a stepper-authored call
  bypassing the executor.
- **The memory file** (`~/.quackd/memory/<adapter>-<backend>.jsonl`). It holds
  sentences a model wrote about a place it has been, it persists between runs, and it is
  read back into the next system prompt. It never leaves the machine and the executor never
  reads it, so a note cannot widen an allowlist, lift a budget or open a confirm gate. What
  would be a security issue: memory reaching the executor, a note from one robot appearing
  in another robot's prompt, or the file escaping the directory `--memory-dir` names.
  `--no-memory` writes nothing at all, and `quackd memory clear` deletes the file.
- **The robot registry** (`~/.quackd/robots.json`, `~/.quackd/flocks.json`). `quackd robot
  add --token ...` writes that token to disk **in plain text**, which is the honest trade for
  not having it in shell history on every run. It is a file in your home directory, not a
  secret store: if that is not good enough for your robot, keep passing `--token` on the line
  or through `QUACKD_DUCK_TOKEN`. quackd masks it in everything it prints, `--json` included,
  which reports only whether one is set. What would be a security issue: a token reaching a
  transcript, a log line, a run directory or an MCP tool result, or either file escaping the
  directory `--registry-dir` names.
- The MCP server executing verbs a loaded `.duck` contract does not allow.
- Anything that lets a `.duck` file (untrusted input — people will share them) execute
  code, read files, or reach the network.
- An adapter sending a body's "go limp" call (`robot.relax`, `disable_motors`,
  `disable_torque`, an XLeRobot `disconnect()`, a ToddlerBot torque-off) as if it were
  `stop`. Stop means stop, never collapse.
- **The browser demo** (`web/`), which is now publicly reachable at
  <https://www.quackd.org/simulator> rather than only a directory you serve yourself. That
  changes the assessment: anybody can be linked to a page that asks them to paste an API key.
  The key is read from an input, sent from the browser straight to the vendor, and never
  stored, never logged and never proxied: there is no server here to proxy it through, and
  nothing in `web/src` writes to browser storage. The vendors a key can reach are exactly the
  entries of `PROVIDERS` in `web/src/providers.js`, each one contacted only once the visitor
  picks it, and the model list offered beside them is generated from the same catalogue the CLI
  ships (`web/src/catalogue.js`, written by `web/build_catalogue.py`), so the page cannot name
  an address or a model this repository does not. What that leaves is the page itself. It
  loads three payloads from `cdn.jsdelivr.net` at pinned versions, plus a stylesheet from
  `fonts.googleapis.com` and the two webfonts it names from `fonts.gstatic.com`, all with no
  subresource integrity and no content security policy, and any script running in the page can
  read that input. Google Fonts serves CSS and font files rather than script, so it cannot
  execute in the document the way the jsDelivr tags can — but it is still an origin that sees
  every visit. So the risk is not quackd holding your key, it is a third party executing in the
  same document as it: a bad CDN response, an injected script, or a copy of the page served
  from somewhere you do not control. <https://www.quackd.org/simulator> is the one copy quackd
  controls, and `/simulator/source.json` names the commit it was built from, so you can check
  it against this repository; the same files served from anywhere else are somebody else's and
  can differ from what is here. Being deployed adds nowhere for a key to be kept: the deployed
  copy is the same static files, fetched into quackd-web's build at a pinned commit, so no
  quackd server sits between the input and the vendor there either. Anthropic's
  `anthropic-dangerous-direct-browser-access` header, which the page sends, is opting out of
  the vendor's own guard against exactly this. Use a key with a spend cap, or pick Local and
  nothing leaves the machine. What the demo cannot do: it is a simulation with no transport to
  any robot, so nothing in it moves hardware.
- **The model and the policies the physics backend fetches** (`adapters/microduck/src/quackd_microduck/sim3d/assets.py`).
  `--robot microduck:mujoco` downloads upstream's MJCF and 38 meshes from codeload.github.com
  and two ONNX policies from huggingface.co, both pinned, and then runs the policy. The defences
  are worth naming because they are the answer: only paths in a fixed allowlist are extracted
  from the tarball, so a crafted archive cannot write outside the cache, and every file is
  checked against a recorded sha256 before MuJoCo or onnxruntime sees it, so a substituted mesh
  or policy fails the run instead of loading. Both are tested rather than merely claimed:
  `tests/test_sim3d_assets.py` builds hostile archives — a traversal path, a sibling directory,
  a symlink, a tampered mesh — and asserts that none of them lands. An ONNX file is data that
  onnxruntime parses, not Python that quackd executes, so the exposure is that parser and not
  arbitrary code. The one
  path around the hashes is deliberate: `QUACKD_MICRODUCK_ASSETS` warns rather than refuses,
  because a newer export from your own checkout is the point of it. Point it at a checkout you
  built, never at one you were sent.
- The LAN surfaces behind `quackd[lan]`: zeroconf TXT records advertise a robot's identity
  to anything on the network, and the MQTT flock bus carries messages that command robots
  with no authentication of its own. Both are off by default and neither has a threat model
  yet, so treat them as trusted-network only.
- **The Jetson container and its compose file** (`deploy/jetson/`,
  [docs/jetson.md](docs/jetson.md)). Three things in there are security decisions rather than
  deployment taste. The model server is pinned to `OLLAMA_HOST: 127.0.0.1:11434` because the
  image's own default is every interface, and under `network_mode: host` that default would put
  a server with no authentication of any kind on whatever network the board is on: reach it from
  a laptop with `ssh -L 11434:127.0.0.1:11434`, the way you reach a robot's bridge. Host
  networking is also what makes the loopback presets true, so the container is on the board's
  loopback rather than isolated from it, and `~/.quackd` is mounted in, which hands the process
  inside the plain text tokens in `robots.json` and every robot's memory file. The compose file
  runs it as uid 1000 rather than as root. That is the access a natively installed quackd already
  has, and it is still worth knowing before you give that mount to an image somebody else built.
  Then `.dockerignore`, which is not housekeeping: the build copies the repository root, a key
  copied into a layer is in the build cache on that machine even after a later layer deletes it,
  and `**/.env` rather than `.env` is what covers `deploy/jetson/.env`, the file the compose file
  reads provider keys from. What would be a security issue: an `.env` reaching a layer, the model
  server binding wider than loopback, or the container being handed anything the compose file
  does not name. None of it has been run on a Jetson, so treat the arrangement as reviewed rather
  than proven.
- **The bridge daemon** (`bridge/open_duck/quackd_duck_bridge.py`), a TCP listener on port
  9871 that walks a 42 cm biped. It binds loopback by default and compares a token with
  `hmac.compare_digest`, but a token is only required if one is configured, and binding it
  wide only warns. Anything that lets an unauthenticated peer move the robot, defeat the
  300 ms deadman, exceed the clamps, or reach past the seven floats the protocol exposes is
  a security issue. Going limp is currently unreachable by construction, and it should stay
  that way: there is no method in the protocol that touches torque.
- **The camera daemon** (`bridge/open_duck/quackd_duck_camd.py`), an HTTP server on port
  9872 that serves a live view of wherever the robot is, with **no authentication at all**.
  It binds loopback by default and warns when told otherwise. Reach it through an ssh
  tunnel. If you bind it wide, everyone on that network can watch your home.
- **The ToddlerBot daemon** (`bridge/toddlerbot/quackd_toddlerbot_bridge.py`), a TCP
  listener on port 9873 that drives a 3 kg humanoid which cannot get up if it falls. It
  binds loopback by default, compares its token with `hmac.compare_digest`, and requires one
  only if one is configured. It is a larger surface than the Open Duck's because it owns the
  50 Hz loop rather than feeding one: anything that lets an unauthenticated peer move the
  robot, defeat the 500 ms deadman, exceed the joint clamps or the per-tick rate limit, or
  reach past the methods the protocol exposes is a security issue. As with the Open Duck,
  going limp is unreachable by construction and must stay that way, and here it matters
  more: torque off on this body means the robot falls over.
- The recommended deployment for all of them is an ssh tunnel
  (`ssh -L 9871:127.0.0.1:9871 -L 9872:127.0.0.1:9872 -L 9873:127.0.0.1:9873`) rather than
  exposing any of these ports.

## Supported versions

Only the latest released minor version receives fixes.

The on-robot artifacts carry their own versions and live on someone else's computer, so
they can drift from the quackd that talks to them: `BRIDGE_VERSION` and `CAMD_VERSION`
on an Open Duck Mini's Raspberry Pi, `VERSION` in the ToddlerBot daemon on its Jetson,
and the AlohaMini host wrapper's `quackd_host_version` field. The two daemons' handshakes carry a
protocol version and refuse a mismatch rather than guessing (the AlohaMini wrapper only
stamps its version into every observation, and quackd does not read it yet), but a daemon
you installed months ago is a daemon that has not had your fixes. `quackd doctor --robot
<adapter>:<backend> --address ...` connects and shows what the robot reports about itself,
though none of these version strings is in that table yet.
