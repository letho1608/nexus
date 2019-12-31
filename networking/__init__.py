# NEXUS Networking Components

# Avoid circular imports by using lazy imports
__all__ = ["Discovery", "PeerHandle", "GRPCServer"]

def __getattr__(name):
    if name == "Discovery":
        from .discovery import Discovery
        return Discovery
    elif name == "PeerHandle":
        from .peer import GRPCPeerHandle as PeerHandle
        return PeerHandle
    elif name == "GRPCServer":
        from .grpc.server import GRPCServer
        return GRPCServer
    raise AttributeError(f"module 'networking' has no attribute '{name}'")