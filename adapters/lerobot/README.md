# quackd-lerobot

The [LeRobot](https://github.com/huggingface/lerobot) SO-101 arm adapter for
[quackd](https://github.com/rokbenko/quackd). Install it through quackd rather than directly:

```bash
uv pip install "quackd[lerobot]"
```

One SO-101 has run this adapter, on 2026-09-15. Every LeRobot name it relies on is read from
upstream source at a pinned commit and listed in `upstream_api.py`.

What it does, what it refuses and why:
[docs/adapters/lerobot.md](https://github.com/rokbenko/quackd/blob/main/docs/adapters/lerobot.md).
