"""
Stateful model utilities for TinyGrad inference engine
"""

class PromptState:
    """Simple prompt state for tracking inference state"""

    def __init__(self, start_pos=0, cache=None):
        self.start = start_pos
        self.cache = cache or []


def make_prompt_state(x, model):
    """
    Create a prompt state for inference

    Args:
        x: Input tensor
        model: Model instance

    Returns:
        PromptState object
    """
    return PromptState(start_pos=0, cache=None)