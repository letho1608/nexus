import os
import sys
import asyncio
from typing import Callable, TypeVar, Optional, Dict, Generic, Tuple, List
import socket
import random
import psutil
import uuid
import re
import subprocess
from pathlib import Path
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor
import traceback

DEBUG = int(os.getenv("DEBUG", default="0"))
DEBUG_DISCOVERY = int(os.getenv("DEBUG_DISCOVERY", default="0"))
VERSION = "0.0.1"

nexus_text = r"""
          _____                    _____                                            _____                    _____          
         /\    \                  /\    \                 ______                   /\    \                  /\    \         
        /::\____\                /::\    \               |::|   |                 /::\____\                /::\    \        
       /::::|   |               /::::\    \              |::|   |                /:::/    /               /::::\    \       
      /:::::|   |              /::::::\    \             |::|   |               /:::/    /               /::::::\    \      
     /::::::|   |             /:::/\:::\    \            |::|   |              /:::/    /               /:::/\:::\    \     
    /:::/|::|   |            /:::/__\:::\    \           |::|   |             /:::/    /               /:::/__\:::\    \    
   /:::/ |::|   |           /::::\   \:::\    \          |::|   |            /:::/    /                \:::\   \:::\    \   
  /:::/  |::|   | _____    /::::::\   \:::\    \         |::|   |           /:::/    /      _____    ___\:::\   \:::\    \  
 /:::/   |::|   |/\    \  /:::/\:::\   \:::\    \  ______|::|___|___ ____  /:::/____/      /\    \  /\   \:::\   \:::\    \ 
/:: /    |::|   /::\____\/:::/__\:::\   \:::\____\|:::::::::::::::::|    ||:::|    /      /::\____\/::\   \:::\   \:::\____\
\::/    /|::|  /:::/    /\:::\   \:::\   \::/    /|:::::::::::::::::|____||:::|____\     /:::/    /\:::\   \:::\   \::/    /
 \/____/ |::| /:::/    /  \:::\   \:::\   \/____/  ~~~~~~|::|~~~|~~~       \:::\    \   /:::/    /  \:::\   \:::\   \/____/ 
         |::|/:::/    /    \:::\   \:::\    \            |::|   |           \:::\    \ /:::/    /    \:::\   \:::\    \     
         |::::::/    /      \:::\   \:::\____\           |::|   |            \:::\    /:::/    /      \:::\   \:::\____\    
         |:::::/    /        \:::\   \::/    /           |::|   |             \:::\__/:::/    /        \:::\  /:::/    /    
         |::::/    /          \:::\   \/____/            |::|   |              \::::::::/    /          \:::\/:::/    /     
         /:::/    /            \:::\    \                |::|   |               \::::::/    /            \::::::/    /      
        /:::/    /              \:::\____\               |::|   |                \::::/    /              \::::/    /       
        \::/    /                \::/    /               |::|___|                 \::/____/                \::/    /        
         \/____/                  \/____/                 ~~                       ~~                       \/____/         
                                                                                                                            
     """

# Single shared thread pool for subprocess operations
subprocess_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="subprocess_worker")


def get_system_info():
  return "Windows"


def find_available_port(host: str = "", min_port: int = 49152, max_port: int = 65535) -> int:
  used_ports_file = os.path.join(tempfile.gettempdir(), "exo_used_ports")

  def read_used_ports():
    if os.path.exists(used_ports_file):
      with open(used_ports_file, "r") as f:
        return [int(line.strip()) for line in f if line.strip().isdigit()]
    return []

  def write_used_port(port, used_ports):
    with open(used_ports_file, "w") as f:
      for p in used_ports[-19:] + [port]:
        f.write(f"{p}\n")

  used_ports = read_used_ports()
  available_ports = set(range(min_port, max_port + 1)) - set(used_ports)

  while available_ports:
    port = random.choice(list(available_ports))
    if DEBUG >= 2: print(f"Trying to find available port {port=}")
    try:
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
      write_used_port(port, used_ports)
      return port
    except socket.error:
      available_ports.remove(port)

  raise RuntimeError("No available ports in the specified range")


def print_nexus():
  print(nexus_text)


def print_yellow_nexus():
  yellow = "\033[93m"  # ANSI escape code for yellow
  reset = "\033[0m"  # ANSI escape code to reset color
  print(f"{yellow}{nexus_text}{reset}")


def terminal_link(uri, label=None):
  if label is None:
    label = uri
  parameters = ""

  # OSC 8 ; params ; URI ST <name> OSC 8 ;; ST
  escape_mask = "\033]8;{};{}\033\\{}\033]8;;\033\\"

  return escape_mask.format(parameters, uri, label)


T = TypeVar("T")
K = TypeVar("K")


class AsyncCallback(Generic[T]):
  def __init__(self) -> None:
    self.condition: asyncio.Condition = asyncio.Condition()
    self.result: Optional[Tuple[T, ...]] = None
    self.observers: list[Callable[..., None]] = []

  async def wait(self, check_condition: Callable[..., bool], timeout: Optional[float] = None) -> Tuple[T, ...]:
    async with self.condition:
      await asyncio.wait_for(self.condition.wait_for(lambda: self.result is not None and check_condition(*self.result)), timeout)
      assert self.result is not None  # for type checking
      return self.result

  def on_next(self, callback: Callable[..., None]) -> None:
    self.observers.append(callback)

  def set(self, *args: T) -> None:
    self.result = args
    for observer in self.observers:
      observer(*args)
    asyncio.create_task(self.notify())

  async def notify(self) -> None:
    async with self.condition:
      self.condition.notify_all()


class AsyncCallbackSystem(Generic[K, T]):
  def __init__(self) -> None:
    self.callbacks: Dict[K, AsyncCallback[T]] = {}

  def register(self, name: K) -> AsyncCallback[T]:
    if name not in self.callbacks:
      self.callbacks[name] = AsyncCallback[T]()
    return self.callbacks[name]

  def deregister(self, name: K) -> None:
    if name in self.callbacks:
      del self.callbacks[name]

  def trigger(self, name: K, *args: T) -> None:
    if name in self.callbacks:
      self.callbacks[name].set(*args)

  def trigger_all(self, *args: T) -> None:
    for callback in self.callbacks.values():
      callback.set(*args)


K = TypeVar('K', bound=str)
V = TypeVar('V')


class PrefixDict(Generic[K, V]):
  def __init__(self):
    self.items: Dict[K, V] = {}

  def add(self, key: K, value: V) -> None:
    self.items[key] = value

  def find_prefix(self, argument: str) -> List[Tuple[K, V]]:
    return [(key, value) for key, value in self.items.items() if argument.startswith(key)]

  def find_longest_prefix(self, argument: str) -> Optional[Tuple[K, V]]:
    matches = self.find_prefix(argument)
    if len(matches) == 0:
      return None

    return max(matches, key=lambda x: len(x[0]))


def is_valid_uuid(val):
  try:
    uuid.UUID(str(val))
    return True
  except ValueError:
    return False


def get_or_create_node_id():
  NODE_ID_FILE = Path(tempfile.gettempdir())/".nexus_node_id"
  try:
    if NODE_ID_FILE.is_file():
      with open(NODE_ID_FILE, "r") as f:
        stored_id = f.read().strip()
      if is_valid_uuid(stored_id):
        if DEBUG >= 2: print(f"Retrieved existing node ID: {stored_id}")
        return stored_id
      else:
        if DEBUG >= 2: print("Stored ID is not a valid UUID. Generating a new one.")

    new_id = str(uuid.uuid4())
    with open(NODE_ID_FILE, "w") as f:
      f.write(new_id)

    if DEBUG >= 2: print(f"Generated and stored new node ID: {new_id}")
    return new_id
  except IOError as e:
    if DEBUG >= 2: print(f"IO error creating node_id: {e}")
    return str(uuid.uuid4())
  except Exception as e:
    if DEBUG >= 2: print(f"Unexpected error creating node_id: {e}")
    return str(uuid.uuid4())


def pretty_print_bytes(size_in_bytes: int) -> str:
  if size_in_bytes < 1024:
    return f"{size_in_bytes} B"
  elif size_in_bytes < 1024**2:
    return f"{size_in_bytes / 1024:.2f} KB"
  elif size_in_bytes < 1024**3:
    return f"{size_in_bytes / (1024 ** 2):.2f} MB"
  elif size_in_bytes < 1024**4:
    return f"{size_in_bytes / (1024 ** 3):.2f} GB"
  else:
    return f"{size_in_bytes / (1024 ** 4):.2f} TB"


def pretty_print_bytes_per_second(bytes_per_second: int) -> str:
  if bytes_per_second < 1024:
    return f"{bytes_per_second} B/s"
  elif bytes_per_second < 1024**2:
    return f"{bytes_per_second / 1024:.2f} KB/s"
  elif bytes_per_second < 1024**3:
    return f"{bytes_per_second / (1024 ** 2):.2f} MB/s"
  elif bytes_per_second < 1024**4:
    return f"{bytes_per_second / (1024 ** 3):.2f} GB/s"
  else:
    return f"{bytes_per_second / (1024 ** 4):.2f} TB/s"


def get_all_ip_addresses_and_interfaces():
    ip_addresses = []
    try:
        # Use socket to get network interface information (Windows compatible)
        hostname = socket.gethostname()
        # Get all IP addresses for this hostname
        try:
            all_ips = socket.getaddrinfo(hostname, None)
            for addr_info in all_ips:
                ip = addr_info[4][0]
                if ip.startswith("0.0.") or ip == "127.0.0.1" or ":" in ip:  # Skip IPv6
                    continue
                ip_addresses.append((ip, f"interface_{len(ip_addresses)}"))
        except:
            pass

        # Also try to get the primary IP
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))  # Connect to external IP to determine local IP
            local_ip = s.getsockname()[0]
            s.close()
            if local_ip not in [ip for ip, _ in ip_addresses]:
                ip_addresses.append((local_ip, "primary"))
        except:
            pass

    except Exception as e:
        if DEBUG >= 1: print(f"Error getting network interfaces: {e}")

    if not ip_addresses:
        if DEBUG >= 1: print("Failed to get any IP addresses. Defaulting to localhost.")
        return [("localhost", "lo")]
    return list(set(ip_addresses))



def get_interface_priority_and_type(ifname: str) -> Tuple[int, str]:
  # Windows interface detection
  if ifname.startswith('lo'):
    return (6, "Loopback")

  # WiFi detection on Windows
  if ifname.startswith(('wlan', 'wifi', 'wl')):
    return (3, "WiFi")

  # Ethernet detection on Windows
  if ifname.startswith(('eth', 'en')):
    return (4, "Ethernet")

  # Virtual interfaces (VPNs, etc.)
  if ifname.startswith(('tun', 'tap', 'ppp')):
    return (1, "External Virtual")

  # Other interfaces
  return (2, "Other")


async def shutdown(signal, loop, server):
  """Gracefully shutdown the server and close the asyncio loop."""
  print(f"Da nhan tin hieu thoat {signal.name}...")
  print("Cam on ban da su dung nexus.")
  print_yellow_nexus()
  server_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
  [task.cancel() for task in server_tasks]
  print(f"Dang huy {len(server_tasks)} tac vu dang cho")
  try:
    await asyncio.gather(*server_tasks, return_exceptions=True)
  except RuntimeError as e:
    if "Event loop is closed" in str(e):
      print("Event loop da dong, bo qua viec huy cac task.")
    else:
      raise
  try:
    await server.stop()
  except RuntimeError as e:
    if "Event loop is closed" in str(e):
      print("Event loop da dong, khong the dung server mot cach nhe nhang.")
    else:
      raise


def is_frozen():
  return getattr(sys, 'frozen', False) or os.path.basename(sys.executable) == "nexus" \
    or '__nuitka__' in globals() or getattr(sys, '__compiled__', False)

def get_nexus_home() -> Path:
  docs_folder = Path(os.environ["USERPROFILE"])/"Documents"
  if not docs_folder.exists(): docs_folder.mkdir(exist_ok=True)
  nexus_folder = docs_folder/"Nexus"
  if not nexus_folder.exists(): nexus_folder.mkdir(exist_ok=True)
  return nexus_folder


def get_nexus_images_dir() -> Path:
  nexus_home = get_nexus_home()
  images_dir = nexus_home/"Images"
  if not images_dir.exists(): images_dir.mkdir(exist_ok=True)
  return images_dir
