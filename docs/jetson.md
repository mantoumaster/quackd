# quackd on an NVIDIA Jetson

A Jetson is the smallest computer that can hold both halves of a run: the model that decides
what the robot should do next, and quackd, which turns that decision into a verb and refuses
the ones the body cannot carry. Put both on the board and a goal in plain language never
leaves the room. That is the whole reason this page exists.

Nothing here has been run on a Jetson by this project. It is written from NVIDIA's own
documentation, and the image is built for arm64 and run under emulation on the machine
that wrote it. The [Status](#status) section at the end says exactly what that does and does not
prove, and what to send back if you run it on yours.

> [!NOTE]
> A Jetson is not a robot. It never appears in `quackd list-adapters`, there is no
> `--robot jetson:...` and there is nothing to install for it. An adapter is a body with a
> manifest and intents; a Jetson is a computer that runs the process, the way your laptop does
> today ([ADR-0044](adr/0044-a-jetson-is-a-host-not-a-body.md)).

## A host, not a body

quackd on a Jetson is the same wheel a laptop installs. It is cross platform Python, it calls
the model over HTTP, it runs the colour detector on the CPU, and nothing in it opens a CUDA
context. The GPU on that board belongs to the **model server**, which is Ollama or
`llama-server` or vLLM, in its own process, reached over loopback through the local presets
that already exist ([local-llms.md](local-llms.md)). So there is no CUDA in quackd's container
and no GPU in its dependency tree, and the interesting engineering is all about the board
rather than about quackd.

Three things about the board are worth knowing before anything else.

**The CPU and the GPU share one pool of memory.** There is no separate VRAM figure to look up.
On an 8 GB Orin Nano the model weights, the KV cache, the operating system, your desktop if you
left one running, and quackd all spend the same 8 GB. It is the reason the model size table
below matters more here than on a desktop with a discrete card.

**`nvidia-smi` is not the tool.** It is the front end for NVML, and NVIDIA's CUDA for Tegra
application note puts NVML under unsupported features: it is supported on Thor and later
only. Use `tegrastats`, whose `GR3D_FREQ` field is the GPU, or `jtop` from `jetson-stats`
for the same numbers with a screen around them.

**One of quackd's seven bodies already carries a Jetson.** A ToddlerBot has one on its back,
and `bridge/toddlerbot/` is the daemon that runs there ([adapters/toddlerbot.md](adapters/toddlerbot.md)).
That is the case where all three things land on one board, and it has a section of its own
below. The Microduck's onboard computer is a Radxa and the Open Duck Mini's is a Raspberry Pi
Zero 2 W, and neither is this.

## Which JetPack

Be on JetPack 6.2 or later in the 6.x line. That is where the wheels, the containers and the
Ollama bundles all exist today.

| L4T | JetPack | Ubuntu | Python | Modules |
|---|---|---|---|---|
| `r39.2.1` | 7.2.1 | 24.04 | 3.12 | Thor and Orin |
| `r39.2.0` | 7.2 | 24.04 | 3.12 | Thor and Orin |
| `r38.4.0` | 7.1 | 24.04 | 3.12 | Thor |
| `r38.2.1` | 7.0 | 24.04 | 3.12 | Thor |
| `r38.2.0` | 7.0 | 24.04 | 3.12 | Thor |
| `r36.5.2` | 6.2.3 | 22.04 | 3.10 | Orin |
| `r36.5.0` | 6.2.2 | 22.04 | 3.10 | Orin |
| `r36.4.4` | 6.2.1 | 22.04 | 3.10 | Orin |
| `r36.4.3` | 6.2 | 22.04 | 3.10 | Orin |
| `r36.4.0` | 6.1 | 22.04 | 3.10 | Orin |
| `r36.3.0` | 6.0 | 22.04 | 3.10 | Orin |

The L4T column is off NVIDIA's JetPack archive and the Ubuntu column off each release's own
download page, both read on 2026-09-22. **NVIDIA publishes no Python version at all**: that
column is Ubuntu's own default `python3` for the release beside it. All of it is version
sensitive by nature, so check it against the board in front of you. `quackd doctor` names your board's JetPack out of this
same table in `quackd/doctor.py`, and `tests/test_deploy_jetson.py` holds the two copies to
each other so the page and the command cannot drift apart. The releases are spelled in three parts here, which is the
form the parser normalises to, so JetPack 6.1 appears as `36.4.0` where NVIDIA's own table
writes `36.4`.

JetPack 5 (L4T 35.x, Ubuntu 20.04, Python 3.8) is below quackd's Python floor of 3.11, so the
container is the route there rather than a native install. JetPack 7 brought the Orin family
onto Ubuntu 24.04 and Python 3.12, which suits quackd better than 6.x does, and its wheel
ecosystem is younger: treat it as forward looking and keep 6.2 as the one this page is
written for.

> [!IMPORTANT]
> JetPack 6 ships Python 3.10 and quackd needs 3.11 or newer. Do not fight the system
> interpreter. `uv` installs its own 3.12 in a second and every command below says so.

## Find out what you have

```bash
cat /etc/nv_tegra_release        # R36 (release), REVISION: 4.3 is L4T r36.4.3
apt-cache show nvidia-jetpack    # the JetPack number itself, which that file never names
cat /proc/device-tree/model      # the board: NVIDIA Jetson Orin Nano Developer Kit
free -h                          # one pool, shared with the GPU
df -h /                          # models are gigabytes each: this wants to be NVMe
sudo tegrastats                  # live: RAM, GR3D_FREQ (the GPU), temperatures, power rails
```

`quackd doctor` answers most of this in one screen once quackd is installed, and it is the
output to paste into an issue.

## Run it natively

The shortest path, and the one to try first. Two installers and one command.

```bash
# uv, which brings its own Python 3.12 and leaves the system 3.10 alone
curl -LsSf https://astral.sh/uv/install.sh -o uv-install.sh
less uv-install.sh && sh uv-install.sh

# Ollama, which reads /etc/nv_tegra_release and fetches the JetPack build for your L4T
curl -fsSL https://ollama.com/install.sh -o ollama-install.sh
less ollama-install.sh && sh ollama-install.sh

ollama pull qwen3:4b             # 8 GB board. On 16 GB or more, qwen3:8b
```

Downloading an installer and reading it before running it is the same stance
`bridge/open_duck/install.sh` takes about itself, and for the same reason: this one writes a
systemd unit and a user on a machine with a robot attached to it.

Then quackd, with no checkout and nothing installed permanently:

```bash
uvx --python 3.12 --from "quackd[openai,microduck]" quackd doctor
uvx --python 3.12 --from "quackd[openai,microduck]" quackd run find-and-kick \
  --llm ollama:qwen3:4b --robot microduck:sim2d
```

> [!IMPORTANT]
> The `openai` extra is the client every local preset speaks through, and `microduck` is a
> robot, because `uv pip install quackd` installs the core and no body at all. `--llm ollama`
> already means `http://localhost:11434/v1`, so there is no address to pass.

> [!NOTE]
> The Jetson section of `doctor` described further down ships in quackd 0.13.0. Until that
> release `uvx` hands you the published wheel, which runs on the board perfectly well and
> simply prints no Jetson block. To see it before then, install from a checkout of `main`.

That run is the cartoon simulator, which needs no robot and no GPU. It is the honest first
test of the board: if the duck kicks the ball, quackd works here, and what is left to find out
is what the model does and how fast the GPU is.

## Which model fits which board

Single user decode on these boards is bound by memory bandwidth, so the practical question is
what fits beside everything else rather than what is fastest.

| Board | Memory | A sensible default |
|---|---|---|
| Orin Nano 8 GB | 8 GB shared | `qwen3:4b`, 2.5 GB on disk |
| Orin NX 16 GB | 16 GB shared | `qwen3:8b`, 5.2 GB on disk |
| AGX Orin 32 GB or 64 GB | 32 GB or 64 GB shared | 8B comfortably, and room to go further |

Download sizes are what Ollama's own library page printed on 2026-09-22 and they are not the
runtime figure: add the KV cache and the runner, and leave headroom for the operating system
and for quackd. **No speed is quoted here on purpose.** Nobody has timed quackd's loop on any
Jetson, and a tokens-per-second number copied from a benchmark of a different prompt on a
different quantisation would be a guess wearing a number's clothes.

The model has to support tool calling, because that is how quackd offers the robot's verbs. If
yours does not, quackd falls back to asking for JSON in the text and retries once, which works
and is worse ([local-llms.md](local-llms.md)). Qwen3 thinks by default, which costs tokens on
every turn; `--extra-body` turns that off and that page shows how.

## Prepare the board

None of this is required to run quackd. All of it matters once a model is resident.

**Put the models and Docker on NVMe.** A microSD card has neither the space nor the read speed,
and a model is loaded from disk every time the server restarts. Move Docker's `data-root` too
if you use the container route.

**Swap, on the NVMe, instead of zram.** JetPack enables zram by default, which compresses RAM
rather than adding any, so it cannot hold what memory could not. `quackd doctor` warns when the
only swap it finds is zram.

```bash
sudo systemctl disable nvzramconfig
sudo fallocate -l 16G /ssd/16GB.swap
sudo chmod 600 /ssd/16GB.swap
sudo mkswap /ssd/16GB.swap && sudo swapon /ssd/16GB.swap
echo '/ssd/16GB.swap none swap sw 0 0' | sudo tee -a /etc/fstab
```

**Give it the power budget.** Mode ids are a property of the flash configuration rather than
of the board, so a number copied out of a blog post is the one thing here that can quietly
do the opposite of what you meant. Read the list instead:

```bash
sudo nvpmodel -q                 # what the modes are on THIS board, and which one is active
sudo nvpmodel -m <id>            # persists across reboots
sudo jetson_clocks               # pins clocks to that mode's maximum; does NOT persist
```

Two traps. `MAXN_SUPER` exists only on an Orin Nano or NX flashed with the super
configuration, so an Orin Nano that was upgraded to JetPack 6.2 rather than reflashed has no
such mode, and `nvpmodel -m 0` there selects 15 W rather than the maximum. And the highest
mode is not automatically the best for a server answering all day: it lifts the power cap
and not the thermal limit, so a board with modest cooling can end up slower than one a step
below it. Pick by the name `nvpmodel -q` prints, not by an id, and measure yours with
`tegrastats` running beside a real workload.

NVIDIA's r36.4 developer guide documents `nvpmodel` but says nothing about whether
`jetson_clocks` survives a reboot. The workaround its own forum recommends is a boot time
unit, so assume it does not.

**Drop the desktop if the board is headless.** `sudo systemctl set-default multi-user.target`
gives the model back whatever the graphical session was holding, and
`graphical.target` puts it back.

## In a container

[`deploy/jetson/`](../deploy/jetson/README.md) holds a Dockerfile and a compose file. The
Dockerfile installs quackd from `uv.lock` at whatever commit you checked out, onto a plain
Debian Python image, and there is no CUDA in it at all.

```bash
git clone https://github.com/rokbenko/quackd && cd quackd/deploy/jetson
mkdir -p runs ~/.quackd                          # Docker would create these owned by root
docker compose --profile ollama up -d            # skip if Ollama is installed natively
docker compose exec ollama ollama pull qwen3:4b
docker compose run --rm quackd doctor
docker compose run --rm quackd run find-and-kick --robot microduck:sim2d
```

Three decisions in that file are worth understanding, because they are the ones people
reasonably expect to be the other way round.

**Ollama gets `runtime: nvidia`, and quackd gets nothing.** The GPU belongs to the model
server. The `deploy.resources.reservations.devices` form that Compose documents for a desktop
card is repeatedly reported not to reach the Tegra integrated GPU, so this file uses the
older `runtime: nvidia` instead. It needs the NVIDIA runtime registered in
`/etc/docker/daemon.json`, and making it the **default** runtime is what NVIDIA's own Jetson
setup documents. That default does not disturb this container: the NVIDIA runtime behaves
exactly like `runc` for an image that sets no `NVIDIA_VISIBLE_DEVICES`, and quackd's sets
none. `quackd doctor` reads the setting back and warns when it is not `nvidia`, but only
when you run it **on the board**: inside the container there is no docker CLI to ask.

**`quackd run` has `restart: "no"` and sits behind a profile.** It is a command: it does one
task, declares a verdict and exits. A restart policy would re-run a robot task every time it
finished and again at every boot, with nobody near the power switch. `docker compose up -d`
with no profile starts nothing at all.

**Both services use host networking.** That is what makes `--llm ollama` mean
`http://localhost:11434/v1` without configuration, and it is also how quackd reaches a robot
daemon running on this same board.

> [!NOTE]
> Only one Ollama. The native installer writes a systemd unit on the same port, so a board set
> up that way must not also start the compose service. There is deliberately no `depends_on`,
> so `docker compose run quackd` cannot start a second one by accident.

## Beside a robot's own daemon

This is the arrangement the board is really for. A ToddlerBot carries a Jetson, and
`bridge/toddlerbot/` is quackd's own daemon for it, owning the fifty hertz control loop that
upstream has none for and listening on port 9873. So one board can hold three processes: the
daemon that moves the robot, the model server, and quackd between them, each reaching the next
over `127.0.0.1`.

Nothing about that is special to quackd. It is the ordinary arrangement with the network hop
shortened to loopback, and every boundary stays where it was: the daemon still owns the body,
quackd still owns the deciding and the gating, and the model still only ever picks one verb.

> [!CAUTION]
> A model server saturating the CPU and the memory bus is exactly the load that can starve a
> fifty hertz control loop, and on a humanoid a starved control loop is a fall. The daemon's
> deadman is what protects the robot there, and it is doing that job for real rather than as a
> formality. Nobody has measured this contention on any board. Put the robot on a stand the
> first time, watch `tegrastats` while a model answers, and consider pinning the model server
> off the cores the loop runs on.

## What quackd doctor shows

On a Jetson, `doctor` grows a section above the usual ones. The block below is what the
renderer actually printed, from a board built out of files by the fixture in
`tests/test_doctor_and_stub.py` with `nvpmodel` and `docker` answered by stubs. Nobody's
hardware produced it, and what a real Orin's own files say is one of the things this page
is asking somebody to send back.

```
· board                   NVIDIA Jetson Orin Nano Developer Kit
· L4T                     36.4.3 (JetPack 6.2)
· memory                  7.3 GiB, 4.9 GiB available, shared with the GPU
⚠ swap                    1.0 GiB, all zram: it compresses RAM rather than adding any
                          (docs/jetson.md)
· GPU device              /dev/nvgpu/igpu0
· power mode              15W (nvpmodel -q)
⚠ docker default runtime  runc: a container is given no GPU unless it is started with
                          --runtime nvidia
```

Two warnings and neither is fatal, which is the point: the whole section is informational and
can never change the exit code. quackd runs fine on that board. What the warnings say is that a
model bigger than memory will not load, and that a container started there will not be given
the GPU.

The section is absent on everything that is not a Tegra, and `--json` carries the same fields
under a `jetson` key, or `null`. Inside a container you will usually see the board but not the
L4T release: `/proc/device-tree` is the host's and is visible from inside, while
`/etc/nv_tegra_release` is a file in the host's root filesystem that a plain Python image does
not have. The servers table below it is the one that answers "is the model up", and it already
probes `localhost:11434` for you.

## Pitfalls

**Ollama answers off the CPU.** `ollama ps` names the processor for a loaded model. The generic
arm64 build has no Tegra CUDA in it; the JetPack build does, and the official installer picks
it by reading `/etc/nv_tegra_release`. If you installed from a tarball by hand, that is the
thing to redo.

**`no kernel image is available for execution on the device`** means a CUDA binary built for a
different GPU architecture. Orin is `sm_87`. Whatever produced that binary needs rebuilding or
replacing with the Jetson artifact.

**llama.cpp fails to allocate with memory apparently free.** It is the unified pool.
`GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` in the server's environment lets an oversized KV cache
spill instead of failing. Build with CUDA on and the architecture set to Orin's, then point
quackd at it with `--llm llamacpp`, which already means port 8080.

**vLLM is not the easy option here.** It is worth it on a big board serving several clients at
once, and on an 8 GB Orin Nano a source build is more likely to exhaust memory than to finish.
For one pilot driving one robot, Ollama or llama.cpp is the right default.

**Rootless Docker and the integrated GPU do not get along.** If the model server is in a
container, run that container under the ordinary root daemon until you have proved otherwise on
your own board.

## Status

**Nothing on this page has been run on a Jetson by this project.** What has been done:

- The image builds for `linux/arm64` and quackd runs inside it, including a whole
  `find-and-kick` task on the scripted pilot. That was done under emulation on the machine
  that wrote this page, and it is the first time quackd itself has run on aarch64 Linux:
  the contributor transcripts in [local-llms.md](local-llms.md) put an aarch64 model server
  behind a quackd that was running on something else.
- `.github/workflows/jetson-image.yml` repeats that on a native arm64 runner with no GPU,
  on every change to these files, so it stays true rather than having been true once.
- The compose file, the extras, and the version table above are held against the code by
  `tests/test_deploy_jetson.py`.
- The JetPack table and the model download sizes were read from NVIDIA's and Ollama's own
  pages on 2026-09-22. Everything version sensitive here goes stale on somebody else's
  release schedule, so check it against the board rather than trusting the date.

What that leaves unproven is everything about the board: Ollama on the Orin GPU, the NVIDIA
container runtime, `nvpmodel`, what the doctor section reads off real files, how much memory a
model actually takes beside quackd, and whether a model server and a robot's control loop can
share one Jetson without the loop suffering.

If you run it, please open a Discussion or an issue with:

- `quackd doctor --json` from the board, which carries the whole `jetson` block
- `runs/<timestamp>-<name>/terminal.txt` and `transcript.jsonl` from a real run
- one `tegrastats` line captured while the model was answering
- `ollama ps` while the model was loaded, and `cat /etc/nv_tegra_release`
- which route you took, native or container, and which board

A transcript is the most useful thing of all, and
[`docs/assets/transcripts/`](assets/transcripts/) is where the contributor ones live.
