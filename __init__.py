# NEXUS - Unified Distributed ML Platform

from .core import Node, Trainer, TrainConfig
from .inference import InferenceEngine, get_inference_engine, Shard
from .training import TrainNode, Strategy, FedAvgStrategy, DiLoCoStrategy
from .networking import Discovery, PeerHandle, GRPCServer
from .topology import Topology, PeerConnection, device_capabilities
from .api import ChatGPTAPI
from .utils import *

__version__ = "0.1.0"
__author__ = "NEXUS Team"

__all__ = [
    "Node", "Trainer", "TrainConfig",
    "InferenceEngine", "get_inference_engine", "Shard",
    "TrainNode", "Strategy", "FedAvgStrategy", "DiLoCoStrategy",
    "Discovery", "PeerHandle", "GRPCServer",
    "Topology", "PeerConnection", "device_capabilities",
    "ChatGPTAPI",
]