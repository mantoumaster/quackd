# quackd on an NVIDIA Jetson, as an arm64 image. See docs/jetson.md.
#
# There is no CUDA in here, and that is the point. quackd is cross platform Python: it calls a
# model over HTTP, runs a detector on the CPU and talks to a robot over a socket. Nothing in
# the core imports a GPU library, the physics extra pins onnxruntime to its CPU provider by
# hand, and the one extra that would want a GPU (`yolo`, which is ultralytics and therefore
# torch, and AGPL besides) is deliberately not installed here. So this is a plain Debian
# Python image that happens to run on a Jetson, and the GPU on that board belongs to the model
# server beside it, which is its own container or its own systemd unit.
#
# That is why it is not built FROM nvcr.io/nvidia/l4t-jetpack. That image is several
# gigabytes of CUDA, cuDNN and TensorRT that nothing here would call, and it ties the result to
# one L4T release. A container shares the host's kernel, so a bookworm userland runs on L4T's
# 5.15 the same way it runs anywhere else.
#
# Build it on the Jetson, or on any arm64 machine, or under emulation:
#   docker buildx build --platform linux/arm64 -f deploy/jetson/Dockerfile -t quackd-jetson .
# The build context is the repository root, because the install comes from `uv.lock` at this
# commit rather than from PyPI. That is what makes the image reproducible and what lets it be
# built from a branch that has not been released.

# The interpreter's minor version and Debian's release are pinned; the patch is not, so a
# rebuild picks up Debian's security updates. Everything Python is pinned exactly by uv.lock,
# which is the half that decides what actually runs.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS build

# uv pinned to the version that wrote uv.lock. `--frozen` below refuses to resolve anything,
# so a uv that read the lock differently would fail loudly rather than quietly building
# something else.
COPY --from=ghcr.io/astral-sh/uv:0.9.30 /uv /bin/uv

# Which bodies and which provider the image carries. `openai` is the client every local preset
# speaks through (Ollama, llama.cpp, vLLM, LM Studio), and `microduck` is a robot, because a
# bare quackd carries none at all. Add one with, for example:
#   --build-arg QUACKD_EXTRAS="--extra openai --extra microduck --extra toddlerbot"
ARG QUACKD_EXTRAS="--extra openai --extra microduck"

ENV UV_PROJECT_ENVIRONMENT=/opt/quackd \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /src
COPY . /src

# `--no-editable` is what makes this copyable: without it uv installs quackd and the adapter as
# links back into /src, and the runtime stage below has no /src. `--no-dev` leaves out pytest,
# ruff and mypy. The venv is built at its final path rather than moved, because a venv that is
# moved after creation has the wrong interpreter in its scripts.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable ${QUACKD_EXTRAS}

FROM ${PYTHON_IMAGE}

# No apt layer at all, and that is a checked fact rather than an oversight. Every wheel here
# is a manylinux aarch64 build carrying what it needs: opencv-python-headless drops the GUI
# and the OpenGL the full package wants, and on bookworm-slim it imports, converts colour,
# encodes a PNG and runs quackd's own detector with nothing added. Checked on arm64 and
# amd64 on 2026-09-22 by building without one and running a whole task.
#
# The CI job runs `import cv2` inside the built image, so a dependency that one day does
# want a system library fails there rather than on somebody's board.

COPY --from=build /opt/quackd /opt/quackd
ENV VIRTUAL_ENV=/opt/quackd \
    PATH=/opt/quackd/bin:$PATH \
    PYTHONUNBUFFERED=1

# Runs land here, and the compose file mounts over it. Owned by uid 1000, which is the first
# user JetPack creates, so a run directory written from inside the container is one you can
# read and delete without sudo. Root can still write it, which is what a bare `docker run`
# does.
RUN install -d -o 1000 -g 1000 /work
WORKDIR /work

LABEL org.opencontainers.image.source="https://github.com/rokbenko/quackd" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="quackd on an NVIDIA Jetson: the CPU half, beside a local model server"

ENTRYPOINT ["quackd"]
# Bare `docker run` says what this image has rather than printing a help screen nobody asked
# for. On a Jetson it also names the board, because /proc/device-tree is the host's.
CMD ["doctor"]
