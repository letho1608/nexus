from abc import ABC, abstractmethod
from typing import Optional
from topology.topology import Topology
from topology.device import DeviceCapabilities

class PeerHandle(ABC):
    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def addr(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def device_capabilities(self) -> DeviceCapabilities:
        pass

    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        pass

    @abstractmethod
    async def disconnect(self):
        pass

    @abstractmethod
    async def send_prompt(self, shard, prompt: str, inference_state: Optional[dict] = None, request_id: Optional[str] = None):
        pass

    @abstractmethod
    async def send_tensor(self, shard, tensor, inference_state: Optional[dict] = None, request_id: Optional[str] = None):
        pass

    @abstractmethod
    async def send_example(self, shard, example, target, length, train: bool, request_id: Optional[str] = None):
        pass

    @abstractmethod
    async def collect_topology(self, visited: set[str], max_depth: int) -> Topology:
        pass

    @abstractmethod
    async def send_result(self, request_id: str, result, is_finished: bool):
        pass

    @abstractmethod
    async def send_opaque_status(self, request_id: str, status: str):
        pass