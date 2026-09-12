# Reading someone else's robot

quackd drives seven bodies and has never run on any of them. Almost everything it does was
worked out by reading upstream code closely enough to be safe without executing it, and the same
handful of traps came up on robot after robot. They are collected here by pattern, because
that is how they recur: the next robot will not have the AlohaMini's bug, it will have a bug
of the AlohaMini's *shape*. The one exception is the Microduck's own model and walking policy,
which `microduck:mujoco` runs on a desktop ([ADR-0030](adr/0030-mujoco-physics-backend.md)), and
executing them taught the last pattern below.

Every claim below is cited at a pinned commit in the relevant adapter's `upstream_api.py`.
This page is the pattern; that file is the evidence.

## A number that looks like a different number

The XLeRobot's wire takes `theta.vel` in **degrees per second**. quackd's `wz` is radians per
second. A pass-through would have been wrong by a factor of 57, in the direction of "faster
than you meant", on a 12 kg cart.

Both ZeroMQ bodies take arm positions in a normalised **−100..100** range, not degrees,
because `use_degrees` defaults to False. The LeRobot adapter, which looks like their sibling,
sets it True and really is in degrees. Copying its `limits` across would have been silent.

And quackd has this trap internally too: its own `look` intent carries a **direction vector**,
which every other body reads with `atan2`. The ToddlerBot's verb takes degrees, and packing
them into the same three slots made a 45 degree sweep arrive as 0.7 degrees.

**What to do:** put the unit in the ref's note, and convert at exactly one place.

## A default pose that is not neutral

Zero is not home. The ToddlerBot's home pose carries plus or minus 1.57 rad of shoulder and
elbow yaw and 1.22 of wrist, so slewing "to zero" is a large motion on every limb, not a small
one. It is also the pose quackd's deadman goes to when nobody is driving.

The same value is overloaded elsewhere: an all-zeros motor reading is exactly what a dropped
packet returns on that bus, so a robot genuinely at zero and a robot you cannot hear look
identical.

**What to do:** read the home pose from the robot, never assume it, and never let a fallback
stand in for it.

## Silence means something different on every body

On a duck, stopping is safe: it stands. On a wheeled base, stopping is safe: it sits still. On
a humanoid mid-stride, "hold the last target forever" and "go limp" are both falls, so its
deadman has to be a *trajectory* — slew to a safe pose and hold — rather than a message.

**What to do:** ask what the body does when nothing arrives, and write the answer into
`safety_authority` rather than assuming zero velocity means stopped.

## A stop that is not a stop

Four of these upstreams disable torque inside `disconnect()`, three by a flag that defaults
on and the ToddlerBot unconditionally. That one also installs a C-level `atexit` handler
that disconnects every client, so **any** normal interpreter exit de-torques a standing
robot, and `SIGTERM` does not even reach it.

**What to do:** treat teardown as part of the stop contract. quackd never calls those, and the
daemon it ships settles to a safe pose before anything is allowed to exit.

## A partial message that means something else

The AlohaMini's driver indexes three velocity keys with no `.get()`, so an action missing one
raises and the **whole** action is discarded, arms included. Its lift is two independent
`if key in action:` blocks with no `else`, so an action carrying neither lift key leaves the
servo travelling while the refreshed watchdog stays quiet.

**What to do:** build one payload, route every verb through it, and test the malformed cases
against a fake that reproduces the bug on purpose.

## A transport that drops your older message

Both ZeroMQ hosts set `CONFLATE`, which keeps only the newest message. Two intents in one tick
become one, silently. quackd's answer is a single writer that re-sends the whole desired
action, so the newest message carries everything.

Related: nothing on either wire is timestamped, and upstream's own client serves its cache on
a read miss — a stopped robot that looks like a moving one. quackd stamps on arrival and turns
a stale reading into a heartbeat failure.

## A capability that is a claim

A command-line flag saying `--camera` is a claim by an operator. If the manifest turns that
into an `observe` verb and nothing behind it can produce a frame, the robot has been made to
lie, and the model will believe it. The same applies to a walk checkpoint that is not staged
and a gripper the build does not have.

**What to do:** report what actually loaded, and narrow the manifest from that. A verb that is
not in the manifest does not exist, which is better than one that exists and refuses.

## A name that exists, on the wrong class

`default_motor_pos` is a real attribute in the ToddlerBot's upstream — on `BasePolicy`, not on
`Robot`, where quackd was reading it. Behind a `getattr(..., default)` it produced a
plausible-looking pose instead of an error.

**What to do:** read the attribute at the pin, and let a rename fail loudly. A fallback on a
safety path converts a typo into a wrong pose.

## A line someone commented out

A stock XLeRobot is blind because every camera entry is commented out. Its host is commented
out of upstream's own package `__init__`. The AlohaMini's arms are limp because both
`enable_torque()` calls are commented out, and its lift keeps travelling after `home()`
because the write that would zero the register is too.

**What to do:** read the config as shipped, not the config as documented. What a robot does
out of the box is what most owners will have.
