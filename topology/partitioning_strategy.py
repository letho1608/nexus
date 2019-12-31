from typing import List
from .topology import Topology

class Partition:
    def __init__(self, node_id: str):
        self.node_id = node_id

    def __repr__(self):
        return f"Partition(node_id='{self.node_id}')"

class PartitioningStrategy:
    def partition(self, topology: Topology) -> List[Partition]:
        """Simple partitioning strategy that creates one partition per node"""
        return [Partition(node_id) for node_id in topology.nodes.keys()]

def map_partitions_to_shards(partitions: List[Partition], n_layers: int, model_id: str):
    """Simple mapping function that distributes layers evenly across partitions"""
    if not partitions:
        return []

    layers_per_partition = n_layers // len(partitions)
    shards = []

    for i, partition in enumerate(partitions):
        start_layer = i * layers_per_partition
        end_layer = start_layer + layers_per_partition - 1 if i < len(partitions) - 1 else n_layers - 1
        shards.append(Shard(model_id, start_layer, end_layer, n_layers))

    return shards

class Shard:
    def __init__(self, model_id: str, start_layer: int, end_layer: int, n_layers: int):
        self.model_id = model_id
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.n_layers = n_layers

    def is_first_layer(self):
        return self.start_layer == 0

    def is_last_layer(self):
        return self.end_layer == self.n_layers - 1

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "start_layer": self.start_layer,
            "end_layer": self.end_layer,
            "n_layers": self.n_layers
        }

    def __repr__(self):
        return f"Shard(model_id='{self.model_id}', start_layer={self.start_layer}, end_layer={self.end_layer}, n_layers={self.n_layers})"