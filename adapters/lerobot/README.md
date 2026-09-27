# quackd-lerobot

The [LeRobot](https://github.com/huggingface/lerobot) SO-101 arm adapter for
[quackd](https://github.com/rokbenko/quackd), and the arm's simulator. Install it through quackd
rather than directly:

```bash
uv pip install "quackd[lerobot]"       # the mock, and the real arm through LeRobot on Python 3.12+
uv pip install "quackd[lerobot-sim]"   # the simulator: MuJoCo and no LeRobot, on Python 3.11+
```

One SO-101 has run this adapter, on 2026-09-15 and again on 2026-09-23. Every LeRobot name it
relies on is read from upstream source at a pinned commit and listed in `upstream_api.py`.

`lerobot:mujoco` is the real backend's own code over a physics model of the SO-101, the maker's
model from TheRobotStudio's SO-ARM100, fetched at a pinned commit on first use and never
shipped. `quackd robot twin` makes one of a registered arm, and `quackd preflight` rehearses task
files on it. Nothing has compared it against an arm. Every name it reads from the model is in
`sim/upstream_api.py`.

What it does, what it refuses and why:
[docs/adapters/lerobot.md](https://github.com/rokbenko/quackd/blob/main/docs/adapters/lerobot.md).
