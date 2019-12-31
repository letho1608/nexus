# NEXUS Training Components

from .train_node import TrainNode
from .strategies.strategy import Strategy
from .strategies.federated_averaging import FedAvgStrategy
from .strategies.diloco import DiLoCoStrategy
from .strategies.communicate_optimize_strategy import CommunicateOptimizeStrategy
from .optimization.optim import OptimSpec, ensure_optim_spec

__all__ = ["TrainNode", "Strategy", "FedAvgStrategy", "DiLoCoStrategy", "CommunicateOptimizeStrategy", "OptimSpec", "ensure_optim_spec"]