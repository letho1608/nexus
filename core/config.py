from dataclasses import dataclass
from typing import Dict, Any, Union, List, Optional, Callable
import torch

# Dummy Strategy class for basic functionality
class Strategy:
    pass

@dataclass
class TrainConfig:
    """Configuration class that holds all training parameters for serialization."""

    model: torch.nn.Module
    train_dataset: Union[
        torch.utils.data.Dataset, Callable[[int, int, bool], torch.utils.data.Dataset]
    ]
    val_dataset: Union[
        torch.utils.data.Dataset, Callable[[int, int, bool], torch.utils.data.Dataset]
    ]
    strategy: Strategy

    num_nodes: int
    rank: Optional[int] = None
    device: Optional[str] = None
    devices: Optional[List[int]] = None
    port: int = 12355

    num_epochs: int = 1 
    max_steps: Optional[int] = None
    batch_size: int = 16
    minibatch_size: int = 16
    shuffle: bool = True
    val_size: int = 64
    val_interval: int = 100
    autocast: bool = False
    checkpoint_interval: Optional[int] = None
    correlation_interval: Optional[int] = None
    save_dir: str = "./checkpoints"
    dataloader_kwargs: Dict[str, Any] = None

    kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.dataloader_kwargs is None:
            self.dataloader_kwargs = {}
