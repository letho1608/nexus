"""
Tokenizers module for NEXUS inference engines
"""

import os
from transformers import AutoTokenizer
from typing import Optional
import asyncio


class NexusTokenizer:
    """Wrapper for transformers tokenizer with async support"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    async def encode(self, text: str) -> list[int]:
        """Async encode text to tokens"""
        if hasattr(self.tokenizer, 'encode') and asyncio.iscoroutinefunction(self.tokenizer.encode):
            # Async tokenizer (like DummyTokenizer)
            return await self.tokenizer.encode(text)
        else:
            # Sync tokenizer (like transformers)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.tokenizer.encode, text)

    async def decode(self, tokens) -> str:
        """Async decode tokens to text"""
        if hasattr(self.tokenizer, 'decode') and asyncio.iscoroutinefunction(self.tokenizer.decode):
            # Async tokenizer (like DummyTokenizer)
            return await self.tokenizer.decode(tokens)
        else:
            # Sync tokenizer (like transformers)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.tokenizer.decode, tokens)

    def apply_chat_template(self, messages, **kwargs):
        """Apply chat template if available"""
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        else:
            # Fallback: just join the messages
            return "\n".join([msg.get("content", "") for msg in messages])


async def resolve_tokenizer(model_path: str, tokenizer_path: Optional[str] = None) -> NexusTokenizer:
    """
    Resolve and load tokenizer for a given model path

    Args:
        model_path: Path to the model directory
        tokenizer_path: Optional specific path to tokenizer

    Returns:
        NexusTokenizer instance
    """
    try:
        # Use tokenizer_path if provided, otherwise use model_path
        tokenizer_dir = tokenizer_path or model_path

        # Try to load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return NexusTokenizer(tokenizer)

    except Exception as e:
        # Fallback: create a minimal dummy tokenizer
        print(f"Warning: Could not load tokenizer from {tokenizer_dir}: {e}")
        print("Using fallback dummy tokenizer")

        class DummyTokenizer:
            def __init__(self):
                self.pad_token = "<pad>"
                self.eos_token = "</s>"

            async def encode(self, text: str) -> list[int]:
                # Simple tokenization: split by spaces and assign dummy IDs
                return [ord(c) % 1000 + 1 for c in text][:512]  # Limit to 512 tokens

            async def decode(self, tokens) -> str:
                # Simple decoding: just return placeholder text
                return f"[Decoded {len(tokens)} tokens]"

            def apply_chat_template(self, messages, **kwargs):
                return "\n".join([msg.get("content", "") for msg in messages])

        return NexusTokenizer(DummyTokenizer())