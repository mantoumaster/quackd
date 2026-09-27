"""A learned policy as the arm's executor: the runners it is asked through and the loop it runs in.

`runner.py` says what a policy is to the arm (`PolicyRunner`, the `Chunk` it answers with), and
`scripted.py` has the runners that need no torch. `loop.py` is the segment `pick` and
`manipulate` hand the arm to. Nothing here imports torch or LeRobot: a checkpoint runs in a
process of its own, never in the one that owns the serial bus.
"""
