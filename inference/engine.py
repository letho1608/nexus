import numpy as np
import os
# from exo.helpers import DEBUG  # Make sure to import DEBUG
import os  # Use os.getenv instead of DEBUG for now

from typing import Tuple, Optional
from abc import ABC, abstractmethod
from .shard import Shard
# from exo.download.shard_download import ShardDownloader  # Comment out for now


class InferenceEngine(ABC):
  session = {}

  @abstractmethod
  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    pass

  @abstractmethod
  async def sample(self, x: np.ndarray) -> np.ndarray:
    pass

  @abstractmethod
  async def decode(self, shard: Shard, tokens: np.ndarray) -> str:
    pass

  @abstractmethod
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    pass

  @abstractmethod
  async def load_checkpoint(self, shard: Shard, path: str):
    pass

  async def save_checkpoint(self, shard: Shard, path: str):
    pass

  async def save_session(self, key, value):
    self.session[key] = value

  async def clear_session(self):
    self.session.clear()

  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    tokens = await self.encode(shard, prompt)
    if shard.model_id != 'stable-diffusion-2-1-base':
      x = tokens.reshape(1, -1)
    else:
      x = tokens
    output_data, inference_state = await self.infer_tensor(request_id, shard, x, inference_state)

    return output_data, inference_state


inference_engine_classes = {
  "tinygrad": "TinygradDynamicShardInferenceEngine",
}


def get_inference_engine(inference_engine_name: str, shard_downloader=None):
  DEBUG = int(os.getenv("DEBUG", default="0"))
  if DEBUG >= 2:
    print(f"get_inference_engine called with: {inference_engine_name}")
  if inference_engine_name == "tinygrad":
    from inference.tinygrad.engine import TinygradDynamicShardInferenceEngine
    from download.manager import NoopShardDownloader
    return TinygradDynamicShardInferenceEngine(NoopShardDownloader())
  else:
    print(f"Unsupported inference engine: {inference_engine_name}")
    return None
