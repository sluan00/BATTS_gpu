"""GPU boosting for two-sample density ratios: a PyTorch port of the boosting part of BATTS."""
from .boosting import BoostFit, Task, boots, boots_many, eval_balance_weight

__version__ = "0.1.0"
__all__ = ["BoostFit", "Task", "boots", "boots_many", "eval_balance_weight"]
