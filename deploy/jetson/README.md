# quackd on an NVIDIA Jetson

Two files. A Dockerfile that puts quackd on an arm64 image with no CUDA in it, and a compose
file that runs a local model beside it on the board's GPU.

| File | What it is |
|---|---|
| `Dockerfile` | quackd, the OpenAI client and the Microduck, installed from `uv.lock` at this commit. Debian, Python 3.12, no CUDA, 395 MB on arm64 |
| `compose.yml` | Ollama on the GPU bound to loopback, and `quackd run` as a one-shot command beside it |

For what a Jetson is to quackd, which JetPack to be on, which model fits which board and how to
run it natively without any of this, see [`docs/jetson.md`](../../docs/jetson.md). That page is
the reference; this one is the two files.

```bash
mkdir -p runs ~/.quackd                          # Docker would create these owned by root
docker compose --profile ollama up -d            # skip if you installed Ollama natively
docker compose exec ollama ollama pull qwen3:4b
docker compose run --rm quackd doctor
docker compose run --rm quackd run find-and-kick --robot microduck:sim2d
docker run --rm --entrypoint python quackd-jetson:local -c 'import cv2'   # the one apt check
```

`doctor` is worth running first. On a Jetson it prints a section naming the board, how much
memory the CPU and GPU are sharing, whether the swap is only zram, and the GPU device node.
Two rows it can only fill in when run on the board rather than in this container: the L4T
release, which lives in a file the image does not have, and Docker's default runtime, which
needs a docker CLI the image does not carry. It says so in both rows rather than leaving
them out.

## The two things people get wrong

**`quackd run` is a command, not a service.** It does one task, declares a verdict and exits.
The compose file gives it `restart: "no"` and hides it behind a profile for that reason: a
restart policy here would re-run a robot task every time it finished and again at every boot.
Ollama is the opposite and is the one service in the file.

**Only one Ollama.** The official installer writes its own systemd unit on the same port, so a
board set up that way must not also start the `ollama` service here. There is no `depends_on`,
so `compose run quackd` cannot start a second one by accident.

## Status

Nothing here has been run on a Jetson by this project. The image was built for arm64 and run under emulation on the
machine that wrote it. `.github/workflows/jetson-image.yml` is set up to repeat that on a
native arm64 Linux runner with no GPU, and it first ran green on 2026-09-23. Between them
they prove the image builds and that quackd runs inside it on aarch64, and they prove
nothing at all about Ollama, the NVIDIA container runtime, or what the board does with a
model loaded. If you run it on yours,
[`docs/jetson.md`](../../docs/jetson.md) says what to send.
