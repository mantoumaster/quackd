# Shoot-day examples for the SO-101 arm

One folder per experiment on the run sheet for the 23 September 2026 shoot, `e001` to `e165`.
Each holds the task file for that experiment, or one per variant, and any picture it hands the
model or asks you to print. A task file's `name` is its experiment number and a slug, so every
run directory says which experiment it was, `runs/20260923-081500-e001-circle/`, and a
`--run-name` on the line tells apart several runs of one file.

Run them from the root of this checkout against the arm registered as `arm-01` with a rest pose
recorded, which is sections 06 and 07 of [lerobot-first-run.md](../lerobot-first-run.md):

```bash
quackd run docs/examples/e001/circle.duck --robot arm-01 --by-hand --camera-url "opencv://1?name=front" --camera-url "opencv://2?name=top" --llm openai:gpt-6-sol --max-steps 16 --no-memory
```

Which cameras, which pilot and whether the arm starts from a pose you set by hand are flags on
the line rather than lines in a file, because the same task runs with one camera or two and with
any model. Each file opens with a comment carrying its first command. Check them all against the
arm's manifest before a session:

```bash
quackd validate "docs/examples/*/*.duck" --robot lerobot:real
```

`e145` and `e152` are the two exceptions: an MCP session loads them from the chat with
`robot_load_duckfile`, which is how their longer budgets reach a session that would otherwise
stop after five minutes.

None of these had run on an arm when they were written. The budgets are estimates, sized so that
a run which goes well finishes well inside them, and the task files are plain text:
[duck-spec.md](../duck-spec.md) is what every field means.
