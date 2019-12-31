from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class RepoProgressEvent:
    """Dummy class for repository download progress tracking"""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RepoProgressEvent':
        """Create instance from dictionary"""
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to dictionary"""
        return {key: getattr(self, key) for key in self.__dict__}