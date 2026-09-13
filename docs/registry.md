# Registered robots

A name for a robot, kept between runs. Not the *verb* registry
([architecture.md](architecture.md)), which is a robot's vocabulary: this is the other half of
the word, a name for a body plus how to reach it.

## Why

Reaching a real robot took five flags, and one of them was a secret:

```bash
quackd run fetch --robot open_duck:bridge --address tcp://10.0.0.5:9871 \
  --token 8f2c... --camera-url http://10.0.0.5:9872/snapshot.jpg
```

Every run retyped them, so the token lived in shell history from then on. `--robots
name=<adapter>:<backend>,...` gave a name, and the name died with the process. Register it
once instead:

```bash
quackd robot add scout open_duck:bridge --address tcp://10.0.0.5:9871 --token 8f2c...
quackd run fetch --robot scout
```

`--robot NAME` works wherever `--robot` does: `run`, `record`, `validate`, `list-verbs`,
`doctor`, `serve-mcp` and `quackd memory`.

## The commands

```bash
quackd robot add NAME <adapter>[:<backend>]   # register one
quackd robot list [--probe]                   # what is registered, and optionally who answers
quackd robot show NAME                        # everything about one, including what it remembers
quackd robot edit NAME [--field X] [--clear F] # change it
quackd robot remove NAME [--force]            # forget it
```

`add` takes `--address`, `--camera-url` and `--token` (the same three flags `run` takes), plus
`--provider` and `--model` for the pilot that drives this robot, and `--note` for a line only
people read.

```
$ quackd robot add duck-a microduck:mock --note "the cream one"
+ added duck-a: microduck:mock
  quackd run <duck> --robot duck-a
```

```
$ quackd robot list
robots (--robot NAME)
+-------------------------------------------------+
| name   | robot          | pilot | note          |
|--------+----------------+-------+---------------|
| arm    | lerobot:mock   | fake  |               |
| duck-a | microduck:mock |       | the cream one |
| duck-b | microduck:mock |       |               |
+-------------------------------------------------+
```

A column nobody has filled is left out, so the table stays readable on a narrow terminal.
`--json` prints one object per line and nothing else, for a script.

```
$ quackd robot show arm
name     arm
robot    lerobot:mock
body     lerobot-so101 (arm, mobility none) 7 verbs: observe, report_state, stop,
         move_joints, gripper, place, pick
address  -
camera   -
token    -
pilot    fake
note     -
flocks   -
added    2026-09-13T13:10:53Z
updated  2026-09-13T13:10:53Z
memory   0 notes, 0 runs  ~/.quackd/memory/arm.jsonl
```

`body` is the robot's static manifest, read without connecting to anything.

## Probing

`--probe` connects to every registered robot, asks how it is, and closes.

```
$ quackd robot list --probe
robots (--robot NAME)
+---------------------------------------------------------------------+
| name   | robot          | pilot | note          | reachable         |
|--------+----------------+-------+---------------+-------------------|
| arm    | lerobot:mock   | fake  |               | + ok              |
| duck-a | microduck:mock |       | the cream one | + ok, battery 88% |
| duck-b | microduck:mock |       |               | + ok, battery 88% |
+---------------------------------------------------------------------+
```

A robot that did not answer shows the reason and the command exits 1, so a script can branch
on it. `--timeout` is per robot and they are probed at once. `microduck:mujoco` is skipped:
connecting to it downloads a model, which is not a liveness check. Every backend named `mock`
always answers, which is what the rows above are.

Nothing here has been pointed at hardware.

## Names

A name is a slug: lowercase letters, digits and hyphens, starting with a letter or digit, 64
characters at most. Three names are refused because they already mean something else:

| Refused | Because |
|---|---|
| `3` | `--flock 3` already means three simulated ducks |
| `microduck` | `--robot microduck` already means `microduck:sim2d` |
| `microduck-sim2d` | that is the memory file an unregistered run of that body already opens |

A colon is what tells the two vocabularies apart: `--robot duck-a` is a name, `--robot
microduck:mock` is a spec, and a bare word that is neither says so in one line.

## Where it lives

| | |
|---|---|
| Directory | `--registry-dir`, else `$QUACKD_REGISTRY_DIR`, else `~/.quackd` |
| Files | `robots.json`, and `flocks.json` for [flocks](flock.md) |
| Format | one JSON object, the name of each robot as its key |

```jsonc
{
  "version": 1,
  "robots": {
    "arm": {
      "spec": "lerobot:mock",
      "address": null,
      "token": null,
      "camera_url": null,
      "provider": "fake",
      "model": null,
      "note": null,
      "added": "2026-09-13T13:10:53Z",
      "updated": "2026-09-13T13:10:53Z"
    }
  }
}
```

It is a file you can open. Each write goes to a temporary file renamed over the old one, so a
reader never sees half of it, and nothing is serialised: two commands writing at the same
instant are a race and the later one wins. That is the same bargain
[memory](memory.md) makes, for the same reason.

Reads are stricter than memory's. A note a model wrote is worth skipping when it is malformed;
a robot's address is not, so an unknown field or a broken file names itself and stops the
command rather than being quietly dropped.

**Tokens are stored in plain text.** `robots.json` is a file in your home directory, not a
secret store. quackd masks the token in everything it prints, including `--json`, which says
only whether one is set. `SECURITY.md` says the same.

## What a name changes

**Memory is keyed by it.** An unregistered run keys its notes by `adapter:backend`, so two
Microducks on one desk shared one file. A registered robot keys by its name, so `duck-a` and
`duck-b` keep separate notes ([memory.md](memory.md),
[ADR-0034](adr/0034-registered-robots-and-pilot-flocks.md)).

**The manifest id is it.** `quackd validate hello-world --robot duck-a` reports the robot as
`duck-a`, which is what `--robots name=spec` has always done for a fleet.

**The pilot can be it.** `quackd robot add scout open_duck:bridge --provider anthropic` means
`quackd run fetch --robot scout` uses Claude without a flag. `--provider` on the line still
wins, and so does `--model`.

**Endpoints come from it.** `--address`, `--token` and `--camera-url` on the line each override
the stored one, field by field, because reaching the same robot through a tunnel today is not
renaming it.

## See also

- [flock.md](flock.md) for `quackd flock` and for running several registered robots at once
- [memory.md](memory.md) for what each robot remembers between runs
- [mcp.md](mcp.md) for `quackd serve-mcp --robot NAME`
- [ADR-0034](adr/0034-registered-robots-and-pilot-flocks.md) for why any of this exists
