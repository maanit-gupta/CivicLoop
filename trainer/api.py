"""Public API of the trainer module. Only this file may be imported by other modules.

run_cycle() runs one learning cycle (collect -> build -> train -> eval -> register -> canary)
and returns the per-module summary. canary_arm(case_id) says which arm serves a case.
"""

from trainer.canary import arm as canary_arm
from trainer.run_cycle import run_cycle

__all__ = ["canary_arm", "run_cycle"]
