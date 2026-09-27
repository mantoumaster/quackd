"""A learned policy as the arm's executor: the runners it is asked through and the loop it runs in.

`runner.py` says what a policy is to the arm (`PolicyRunner`, the `Chunk` it answers with), and
`scripted.py` has the runners that need no torch. `loop.py` is the segment `pick` and
`manipulate` hand the arm to. Nothing here imports torch or LeRobot: a checkpoint runs in a
process of its own, never in the one that owns the serial bus. `server.py` is that process
(`quackd policy serve`), `client.py` is how the arm reaches it (`RemoteRunner`), `protocol.py`
is what the two say to each other, and `upstream_api.py` is every LeRobot policy name either
relies on.
"""
