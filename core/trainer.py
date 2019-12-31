import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# from exogym.train_node import TrainNode
# from exogym.strategy import Strategy
# from exogym.common import TrainConfig
# from exogym.aux.utils import print_dataset_size, _average_model_states
# from exogym.minibatch_probe import find_minibatch_size_isolated
# from exogym.utils import init_process_group_portsafe

# Dummy classes for basic functionality
class TrainNode:
    def __init__(self, config):
        pass
    def train(self):
        return {}

class Strategy:
    def _init_node(self, model, rank, num_nodes):
        pass

import os
import copy
from dataclasses import dataclass
from typing import Optional, List, Any, Dict, Union, Callable
from collections import OrderedDict

@dataclass
class TrainConfig:
    model: Any = None
    train_dataset: Any = None
    val_dataset: Any = None
    strategy: Strategy = None
    num_epochs: int = 1
    num_nodes: int = 1
    max_steps: Optional[int] = None
    port: int = 12355
    device: str = "cpu"
    devices: Optional[List[int]] = None
    batch_size: int = 16
    minibatch_size: Optional[int] = None
    shuffle: bool = True
    val_size: int = 64
    val_interval: int = 100
    autocast: bool = False
    checkpoint_interval: Optional[int] = None
    correlation_interval: Optional[int] = None
    save_dir: str = "./checkpoints"
    dataloader_kwargs: Dict[str, Any] = None
    kwargs: Dict[str, Any] = None
    rank: int = 0

def print_dataset_size(*args):
    pass

def _average_model_states(model_states):
    return {}

def find_minibatch_size_isolated(*args):
    return 16

def init_process_group_portsafe(*args):
    pass


def _build_connection(config: TrainConfig):
    """
    This is the default callback for setting up pytorch distributed connections.
    All ranks are assumed to be on the same machine, and device is defaulted to cpu.
    In future, this can be swapped out assuming non-localhost connections, etc.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(config.port)

    if config.device == "" or config.device is None:
        if torch.cuda.is_available():
            config.device = "cuda"
        elif torch.backends.mps.is_available():
            config.device = "mps"
        else:
            config.device = "cpu"

    # initialize the process group
    if config.device == "cuda":
        # If we haven't specified devices, use all devices.
        if config.devices is None:
            config.devices = range(torch.cuda.device_count())

        init_process_group_portsafe(
            "nccl" if len(config.devices) == config.num_nodes else "gloo",
            rank=config.rank,
            world_size=config.num_nodes,
        )
        config.device = torch.device(
            f"cuda:{config.devices[config.rank % len(config.devices)]}"
        )
        torch.cuda.set_device(config.device)
    elif config.device == "cpu":
        init_process_group_portsafe("gloo", rank=config.rank, world_size=config.num_nodes)
        config.device = torch.device("cpu")
    elif config.device == "mps":
        init_process_group_portsafe("gloo", rank=config.rank, world_size=config.num_nodes)
        config.device = torch.device("mps")
    else:
        raise ValueError(f"Invalid device type: {config.device}")

    print(f"Rank {config.rank} using device {config.device}")

def _worker(rank: int, config: TrainConfig, result_queue: mp.Queue):
    """
    Entry point executed in every child process.
    This function is importable as exogym.trainer._worker, making it notebook-safe.
    """
    config.rank = rank    

    _build_connection(config)

    # TODO: Should these happen here or in TrainNode.__init__() ?
    config.model = copy.deepcopy(config.model).to(config.device)
    config.strategy = copy.deepcopy(config.strategy)
    config.strategy._init_node(config.model, config.rank, config.num_nodes)

    train_node = TrainNode(config=config)
    final_model_state = train_node.train()

    # Move tensors to CPU and detach to avoid CUDA serialization issues
    cpu_state_dict = OrderedDict()
    for key, tensor in final_model_state.items():
        cpu_state_dict[key] = tensor.detach().cpu()

    result_queue.put((rank, cpu_state_dict))

    dist.destroy_process_group()

class Trainer:
    """
    Trainer is used to train a model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Union[
            torch.utils.data.Dataset,
            Callable[[int, int, bool], torch.utils.data.Dataset],
        ],
        val_dataset: Union[
            torch.utils.data.Dataset,
            Callable[[int, int, bool], torch.utils.data.Dataset],
        ],
        start_port: Optional[int] = None,
        device: str = None,
        devices: list[int] = None,
    ):
        self.model_orig = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.port = start_port if start_port is not None else 12355
        self.device = device
        self.devices = devices
        self._minibatch_cache = {}

    def fit(
        self,
        num_epochs: int,
        strategy: Strategy,
        num_nodes: int,
        max_steps: int = None,
        batch_size: int = 16,
        minibatch_size: int = None,
        shuffle: bool = True,
        val_size: int = 64,
        val_interval: int = 100,
        autocast: bool = False,
        checkpoint_interval: Optional[int] = None,
        correlation_interval: Optional[int] = None,
        save_dir: str = "./checkpoints",
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # assert val_size // batch_size > 0, f"val_size must be geq batch_size: {val_size} // {batch_size}"
        assert batch_size > 0, 'local batch size needs to be nonzero'
        if minibatch_size is not None:
            assert minibatch_size <= batch_size, f'minibatch_size ({minibatch_size}) must be <= batch_size ({batch_size}) for gradient accumulation'

        # Move a *copy* of the model to CPU so that pickling for mp.spawn does not attempt to share GPU storage.
        cpu_model = copy.deepcopy(self.model_orig).cpu()

        self.config = TrainConfig(
            model=cpu_model,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            strategy=strategy,
            num_epochs=num_epochs,
            num_nodes=num_nodes,
            max_steps=max_steps,
            port=self.port,
            device=self.device,
            devices=self.devices,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            shuffle=shuffle,
            val_size=val_size,
            val_interval=val_interval,
            autocast=autocast,
            checkpoint_interval=checkpoint_interval,
            correlation_interval=correlation_interval,
            save_dir=save_dir,
            dataloader_kwargs=dataloader_kwargs or {},
            kwargs=kwargs,
        )

        # Auto-detect minibatch_size if not provided
        if minibatch_size is None:
            force_recalculate = kwargs.get('force_minibatch_recalculate', False)
            minibatch_size = self.find_minibatch_size(
                num_nodes,
                batch_size,
                force_recalculate=force_recalculate,
            )
            self.config.minibatch_size = minibatch_size

        self.port += 1

        
        manager = mp.Manager()
        result_queue = manager.Queue()

        mp.spawn(
            _worker,
            args=(self.config, result_queue),
            nprocs=self.config.num_nodes,
            start_method="spawn",
            join=True,
        )

        model_states = {}
        for _ in range(self.config.num_nodes):
            rank, state_dict = result_queue.get()
            model_states[rank] = state_dict

        averaged_state_dict = _average_model_states(model_states)

        final_model = copy.deepcopy(self.model_orig)
        final_model.load_state_dict(averaged_state_dict)
        return final_model

    def clear_minibatch_cache(self):
        """Clear the cached minibatch size results."""
        self._minibatch_cache.clear()
        print("Minibatch size cache cleared.")
    
    def find_minibatch_size(self, num_nodes: int, batch_size: int, force_recalculate: bool = False):
        cache_key = (num_nodes, batch_size)
        
        if not force_recalculate and cache_key in self._minibatch_cache:
            cached_size = self._minibatch_cache[cache_key]
            print(f'Using cached minibatch_size={cached_size} for batch_size={batch_size}, num_nodes={num_nodes}')
            return cached_size
        
        # Use the isolated minibatch probe
        minibatch_size = find_minibatch_size_isolated(
            self.config,
            num_nodes,
            batch_size,
            devices=self.devices,
            device=self.device,
            port=self.port
        )
        
        self._minibatch_cache[cache_key] = minibatch_size
        return minibatch_size 
