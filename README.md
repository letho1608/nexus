# NEXUS - Hệ thống suy luận phân tán trên Windows

**NEXUS** là một nền tảng Machine Learning phân tán được tối ưu hóa cho Windows, tập trung vào khả năng suy luận (inference) với kiến trúc đơn giản và hiệu quả.

## 🚀 Tính năng chính

### **Suy luận phân tán**
- **TinyGrad Engine**: Sử dụng TinyGrad làm engine suy luận duy nhất, được tối ưu hóa cho Windows
- **Model Sharding**: Tự động phân chia model trên nhiều nodes Windows
- **Dynamic Model Loading**: Tải model từ HuggingFace hoặc local với định dạng TinyGrad
- **ChatGPT API Compatible**: Thay thế hoàn toàn OpenAI API

### **Networking & Discovery**
- **gRPC Communication**: Giao tiếp hiệu năng cao giữa các nodes
- **UDP Discovery**: Tự động phát hiện nodes qua giao thức UDP
- **Topology Management**: Quản lý topology mạng động với theo dõi khả năng thiết bị

### **Web Interface**
- **Real-time Chat**: Giao diện chat tương tác
- **Topology Visualization**: Trực quan hóa topology mạng
- **API Monitoring**: Giám sát hoạt động API thời gian thực

## 📁 Cấu trúc dự án

```
nexus/
├── core/                    # Thành phần cốt lõi
│   ├── node.py             # Node suy luận phân tán
│   ├── trainer.py          # Engine training (placeholder)
│   ├── config.py           # Quản lý cấu hình
│   └── models.py           # Quản lý mô hình
├── inference/              # Khả năng suy luận
│   ├── engine.py           # Base inference engine
│   ├── shard.py            # Model sharding
│   ├── tokenizers.py       # Xử lý tokenizer
│   └── tinygrad/          # TinyGrad inference engine
│       ├── engine.py       # TinyGrad inference engine
│       ├── helpers.py      # Helper functions
│       ├── losses.py       # Loss functions
│       ├── models/         # Model implementations
│       └── stateful_model.py # Stateful model wrapper
├── training/               # Khả năng training (placeholder)
│   ├── train_node.py       # Training node
│   ├── strategies/        # Training strategies
│   └── auxiliary/         # Training utilities
├── networking/             # Networking stack
│   ├── grpc/              # gRPC server
│   ├── discovery.py       # Service discovery
│   ├── peer.py            # Peer management
│   └── udp_discovery.py   # UDP discovery
├── topology/               # Quản lý topology
│   ├── topology.py        # Network topology
│   ├── device.py          # Device capabilities
│   └── partitioning_strategy.py # Partitioning strategy
├── download/               # Download management
│   ├── manager.py          # Shard download manager
│   └── download_progress.py # Progress tracking
├── api/                    # API layer
│   └── chat.py            # ChatGPT-compatible API
├── web/                    # Web interface
│   ├── index.html         # Main web interface
│   ├── index.js           # Frontend JavaScript
│   ├── index.css          # Styles
│   └── static/            # Static assets
├── utils/                  # Shared utilities
│   ├── helpers.py         # Helper functions
│   └── topology_viz.py    # Topology visualization
├── main.py                # Main entry point
└── README.md              # This file
```

## 🛠️ Cài đặt và sử dụng

### **Yêu cầu hệ thống**
- **Windows 10/11** (chỉ hỗ trợ Windows)
- Python 3.10+
- TinyGrad (cho suy luận CPU/GPU)
- Các thư viện hệ thống Windows

### **Cài đặt**
```bash
# Clone repository
git clone github.com/letho1608/nexus
cd nexus

# Install dependencies
pip install -r requirements.txt

# Khởi tạo
python main.py --help
```

### **Sử dụng cơ bản**

#### **1. Chạy suy luận với mô hình test**
```bash
python main.py run test-model
```

#### **2. Chạy suy luận với mô hình TinyGrad**
```bash
python main.py run llama-7b
```

#### **3. Chạy với debug mode**
```bash
set DEBUG=1 && python main.py run llama-7b
```

#### **4. Cấu hình nâng cao**
```bash
python main.py run llama-7b \
  --node-host 0.0.0.0 \
  --api-port 8000 \
  --max-tokens 2048 \
  --temperature 0.7
```

## 🔧 Cấu hình

### **Node Configuration**
```python
config = NexusConfig(
    node_id="my-node-001",
    node_host="0.0.0.0",
    node_port=5678,
    discovery_method="udp",
    wait_for_peers=2,
)
```

### **Inference Configuration**
```python
config = NexusConfig(
    inference_engine="tinygrad",  # Chỉ hỗ trợ TinyGrad
    default_model="llama-7b",
    max_generate_tokens=2048,
    default_temperature=0.0,
)
```

### **Training Configuration (Experimental)**
```python
config = NexusConfig(
    num_epochs=10,
    batch_size=32,
    minibatch_size=8,
    training_strategy="federated_averaging",
    checkpoint_dir="./checkpoints",
)
```

## 🌐 API Usage

### **ChatGPT-Compatible API**
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:52415/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="llama-2-7b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### **Direct Node API**
```python
from nexus.core import Node

node = Node(...)
result = await node.process_prompt(shard, "Your prompt here")
```


## 🧪 Testing

Hiện tại chưa có bộ test tự động. Testing sẽ được bổ sung trong các phiên bản tương lai.

## 📊 Monitoring & Debugging

### **Debug Levels**
```python
import os
os.environ["DEBUG"] = "2"  # 0=quiet, 1=normal, 2=verbose
```

**NEXUS** - Hệ thống suy luận phân tán TinyGrad được tối ưu hóa cho Windows! 🚀