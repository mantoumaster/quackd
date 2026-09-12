"""quackd — the brain daemon Microduck was missing, now a brain for any small robot.

This package exists so that any LLM can pilot a small robot (a Microduck, an arm, a
wheeled base; real or simulated) through a safety-enforced vocabulary of *verbs*
that the robot's own manifest declares, driven by a `.duck` skill file. The LLM picks
verbs, the robot's own controllers move, quackd enforces the contract. Everything else in
the repo serves that sentence.
"""

__version__ = "0.8.0"
