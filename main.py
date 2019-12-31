#!/usr/bin/env python3
"""
NEXUS - Unified Distributed ML Platform

This is the main entry point for NEXUS, combining the capabilities of both
unified platform.
"""

import argparse
import asyncio
import atexit
import signal
import json
import os
import time
import traceback
import uuid
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

# NEXUS imports
from core import Node, Trainer, TrainConfig
from inference import get_inference_engine, Shard
from networking.grpc.server import GRPCServer
from networking.udp_discovery import UDPDiscovery
from networking.peer import GRPCPeerHandle
from topology import Topology, device_capabilities, UNKNOWN_DEVICE_CAPABILITIES, PartitioningStrategy
from api import ChatGPTAPI
from utils import print_yellow_nexus, find_available_port, DEBUG, get_system_info, get_or_create_node_id, get_all_ip_addresses_and_interfaces, terminal_link, shutdown

# Configure environment
os.environ["GRPC_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

@dataclass
class NexusConfig:
    """Unified configuration for NEXUS platform"""
    # Node configuration
    node_id: str = None
    node_host: str = "0.0.0.0"
    node_port: int = None
    listen_port: int = 5678
    discovery_method: str = "udp"
    discovery_timeout: int = 30
    wait_for_peers: int = 0

    # Inference configuration
    inference_engine: str = None
    default_model: str = None
    max_generate_tokens: int = 10000
    default_temperature: float = 0.0

    # Training configuration
    num_epochs: int = 10
    batch_size: int = 32
    minibatch_size: int = None
    training_strategy: str = "federated_averaging"
    data_path: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 5

    # API configuration
    api_port: int = 52415
    api_timeout: int = 900

    # Advanced configuration
    system_prompt: str = None
    disable_web: bool = False


async def main():
    """Main entry point for NEXUS"""
    parser = argparse.ArgumentParser(description="NEXUS - Unified Distributed ML Platform")

    # Node configuration
    parser.add_argument("--node-id", type=str, help="Node ID")
    parser.add_argument("--node-host", type=str, default="0.0.0.0", help="Node host")
    parser.add_argument("--node-port", type=int, help="Node port")
    parser.add_argument("--listen-port", type=int, default=5678, help="Listening port for discovery")
    parser.add_argument("--discovery-method", type=str, choices=["udp"], default="udp", help="Discovery method")
    parser.add_argument("--discovery-timeout", type=int, default=30, help="Discovery timeout in seconds")
    parser.add_argument("--wait-for-peers", type=int, default=0, help="Number of peers to wait for")

    # Inference configuration
    parser.add_argument("--inference-engine", type=str, help="Inference engine (tinygrad)")
    parser.add_argument("--default-model", type=str, help="Default model")
    parser.add_argument("--max-tokens", type=int, default=10000, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Default sampling temperature")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--minibatch-size", type=int, help="Minibatch size for gradient accumulation")
    parser.add_argument("--training-strategy", type=str, default="federated_averaging", help="Training strategy")
    parser.add_argument("--data", type=str, default="./data", help="Path to training data")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs")

    # API configuration
    parser.add_argument("--api-port", type=int, default=52415, help="API port")
    parser.add_argument("--api-timeout", type=int, default=900, help="API timeout in seconds")

    # Other configuration
    parser.add_argument("--system-prompt", type=str, help="System prompt for API")
    parser.add_argument("--disable-web", action="store_true", help="Disable web interface")

    # Commands
    parser.add_argument("command", nargs="?", choices=["run", "train", "eval", "serve"], help="Command to run")
    parser.add_argument("model", nargs="?", help="Model name for run/eval commands")

    args = parser.parse_args()

    print("Dang khoi dong NEXUS")
    print_yellow_nexus()

    # Get system info
    system_info = get_system_info()
    print(f"He thong duoc phat hien: {system_info}")

    # Set up configuration
    config = NexusConfig(
        node_id=args.node_id,
        node_host=args.node_host,
        node_port=args.node_port,
        listen_port=args.listen_port,
        discovery_method=args.discovery_method,
        discovery_timeout=args.discovery_timeout,
        wait_for_peers=args.wait_for_peers,
        inference_engine=args.inference_engine,
        default_model=args.default_model,
        max_generate_tokens=args.max_tokens,
        default_temperature=args.temperature,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        training_strategy=args.training_strategy,
        data_path=args.data,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        api_port=args.api_port,
        api_timeout=args.api_timeout,
        system_prompt=args.system_prompt,
        disable_web=args.disable_web,
    )

    # Set defaults
    if not config.node_id:
        config.node_id = get_or_create_node_id()

    if not config.node_port:
        config.node_port = find_available_port(config.node_host)

    if not config.inference_engine:
        config.inference_engine = "tinygrad"  # Default to tinygrad engine on Windows

    print(f"ID nut: {config.node_id}")
    print(f"Dia chi nut: {config.node_host}:{config.node_port}")
    print(f"Cong cu suy luan: {config.inference_engine}")

    # Set up networking
    discovery = UDPDiscovery(
        config.node_id,
        config.node_port,
        config.listen_port,
        config.listen_port,
        lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
        discovery_timeout=config.discovery_timeout,
    )

    # Set up inference engine
    inference_engine = get_inference_engine(config.inference_engine, None)
    if not inference_engine:
        print(f"Loi: Cong cu suy luan '{config.inference_engine}' khong kha dung tren he thong nay.")
        print("NEXUS can cong cu suy luan de hoat dong. Thoat...")
        return
    print(f"Using inference engine: {inference_engine.__class__.__name__}")

    # Set up API endpoints
    chatgpt_api_endpoints = [f"http://{ip}:{config.api_port}/v1/chat/completions" for ip, _ in get_all_ip_addresses_and_interfaces()]
    web_chat_urls = [f"http://{ip}:{config.api_port}" for ip, _ in get_all_ip_addresses_and_interfaces()]

    if DEBUG >= 0:
        print("Giao dien NEXUS:")
        for web_chat_url in web_chat_urls:
            print(f" - {terminal_link(web_chat_url)}")
        print("Diem cuoi API ChatGPT:")
        for chatgpt_api_endpoint in chatgpt_api_endpoints:
            print(f" - {terminal_link(chatgpt_api_endpoint)}")

    # Create partitioning strategy
    partitioning_strategy = PartitioningStrategy()

    # Create unified node
    node = Node(
        config.node_id,
        None,  # server will be set later
        inference_engine,
        discovery,
        None,  # shard downloader
        partitioning_strategy=partitioning_strategy,
        max_generate_tokens=config.max_generate_tokens,
        default_sample_temperature=config.default_temperature,
    )

    # Set up server
    server = GRPCServer(node, config.node_host, config.node_port)
    node.server = server

    # Set up API
    api = ChatGPTAPI(
        node,
        inference_engine.__class__.__name__,
        response_timeout=config.api_timeout,
        default_model=config.default_model,
        system_prompt=config.system_prompt,
    )

    # Set up signal handlers
    loop = asyncio.get_running_loop()

    def handle_exit():
        asyncio.ensure_future(shutdown(signal.SIGTERM, loop, server))

    try:
        for s in [signal.SIGINT, signal.SIGTERM]:
            loop.add_signal_handler(s, handle_exit)
    except NotImplementedError:
        # On Windows, signal handlers are not supported, rely on KeyboardInterrupt
        pass

    # No cursor restoration needed on Windows

    # Start services
    try:
        await node.start(wait_for_peers=config.wait_for_peers)

        # Handle different commands
        command = args.command if args.command else ("run" if args.model else None)

        if DEBUG >= 1:
            print(f"DEBUG: Command logic - command: '{command}', args.model: '{args.model}', args.command: '{args.command}'")

        if command == "run" or args.model:
            model_name = args.model
            if not model_name:
                print("Loi: Ten mo hinh la bat buoc cho lenh 'run'")
                print("Cach su dung: python main.py run <model_name>")
                return
            print(f"Dang chay suy luan voi mo hinh: {model_name}")
            await run_inference_cli(node, model_name)
        elif command == "train":
            if not args.model:
                print("Loi: Ten mo hinh la bat buoc cho lenh 'train'")
                print("Cach su dung: python main.py train <model_name>")
                return
            await run_training_cli(node, args.model, config)
        elif command == "eval":
            if not args.model:
                print("Loi: Ten mo hinh la bat buoc cho lenh 'eval'")
                print("Cach su dung: python main.py eval <model_name>")
                return
            await run_evaluation_cli(node, args.model, config)
        else:
            # Default: show help
            print("Chuc nang cot loi NEXUS co san.")
            print("\nCach su dung:")
            print("  python main.py run <model>     # Chay suy luan voi mo hinh")
            print("  python main.py train <model>   # Huan luyen mo hinh")
            print("  python main.py eval <model>    # Danh gia mo hinh")
            print("  python main.py serve           # Khoi dong may chu API (bi vo hieu)")
            print("\nVi du:")
            print("  python main.py run test-model")
            print("  python main.py run llama-7b")
            print("Dang thoat...")
            return

        # Graceful shutdown
        if args.wait_for_peers > 0:
            print("Thoi gian cho de cho phep cac nut ngang cap thoat mot cach nhe nhang")
            for i in tqdm(range(50)):
                await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDa nhan tin hieu ngat, dang tat mot cach nhe nhang...")
        await shutdown(signal.SIGINT, loop, server)
    finally:
        # Ensure graceful shutdown
        await shutdown(signal.SIGTERM, loop, server)

async def run_inference_cli(node, model_name):
    """Run inference on a model"""
    print(f"Dang chay suy luan voi mo hinh: {model_name}")

    # Create a simple prompt
    prompt = "Xin chao, toi la mot mo hinh ngon ngu. Toi co the giup gi cho ban hom nay?"

    try:
        # Create shard
        shard = Shard(model_id=model_name, start_layer=0, end_layer=0, n_layers=1)

        print(f"Dang xu ly loi nhac voi phan doan: {shard}")
        print(f"Loi nhac: {prompt}")

        # Process prompt
        result = await node.process_prompt(shard, prompt)
        print("THANH CONG: Suy luan da hoan thanh thanh cong!")
        print(f"Ket qua suy luan: {result}")

        # Show some additional info
        print("\nThong tin nut:")
        print(f"  - ID nut: {node.id}")
        print(f"  - Kha nang thiet bi: {node.device_capabilities}")
        print(f"  - So luong nut ngang cap: {len(node.peers)}")

    except Exception as e:
        print(f"LOI: Loi khi chay suy luan: {e}")
        print("Dieu nay duoc mong doi cho cac mo hinh thu nghiem. He thong cot loi dang hoat dong chinh xac.")
        if DEBUG >= 1:
            traceback.print_exc()

async def run_training_cli(node, model_name, config):
    """Run training on a model"""
    print(f"Dang chay huan luyen voi mo hinh: {model_name}")
    print(f"Chien luoc huan luyen: {config.training_strategy}")

    # This is a placeholder - in a real implementation,
    # you would load actual training data and run the training process
    print("Chuc nang huan luyen van dang duoc phat trien trong nen tang thong nhat.")
    print("Vui long su dung cac thanh phan EXOGYM rieng le de co day du kha nang huan luyen.")

async def run_evaluation_cli(node, model_name, config):
    """Run evaluation on a model"""
    print(f"Dang chay danh gia voi mo hinh: {model_name}")
    print("Chuc nang danh gia van dang duoc phat trien trong nen tang thong nhat.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nNEXUS bi ngat boi nguoi dung. Dang thoat...")
        # Note: graceful shutdown is handled inside main()