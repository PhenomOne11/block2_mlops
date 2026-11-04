# Low-Latency Containerized CIFAR-10 Inference Microservice with Asynchronous Dynamic Batching

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.18.0_INT8-blue.svg?style=flat&logo=onnx)](https://onnxruntime.ai)
[![Docker](https://img.shields.io/badge/Docker-Multi--Stage_Build-2496ED.svg?style=flat&logo=docker)](https://www.docker.com)
[![Locust](https://img.shields.io/badge/Load_Testing-Locust_2.47-green.svg?style=flat&logo=locust)](https://locust.io)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python)](https://www.python.org)

An enterprise-grade, resource-constrained asynchronous inference microservice engineered for high-concurrency CIFAR-10 image classification. Built from scratch with dynamic micro-batching, INT8 graph quantization, Prometheus observability, and hardened non-root container isolation.

---

## 1. System Architecture & High-Concurrency Data Flow

Standard inference deployments process HTTP calls synchronously, causing thread starvation and high resource contention under concurrent bursts. This system implements an asynchronous producer-consumer ticketing pattern:

```
[Concurrent Clients] 
   │
   ▼
[HTTP Ingress / UVICORN Event Loop]
   │
   ├──> Pydantic v2 Base64 / Array Serialization Guard
   │
   ├──> Push to asyncio.Queue (Async Request Ingestion)
   │     │
   │     ▼
┌────────────────────────────────────────────────────────┐
│               Asynchronous Batcher Engine              │
│                                                        │
│  - Collects requests up to MAX_BATCH_SIZE (32)         │
│  - Or drains on timeout trigger MAX_DELAY_MS (10ms)    │
│  - Stacks tensors along N-dimension: (N, 3, 32, 32)    │
└────────────────────────────────────────────────────────┘
   │
   ▼
[ONNX Runtime Engine - INT8 Quantized Core]
   │
   ├──> Dynamic Thread-Pooled Vectorized Execution (AVX-512 / VNNI)
   │
   ▼
[Result Dispatcher via asyncio.Future Event Loop Uncoupling]
   │
   ├──> Classify ArgMax & Softmax Confidence
   │
   ▼
[HTTP 200 OK Response Returned to Individual Sockets]
```

---

## 2. Key Engineering Innovations

### Dynamic Micro-Batching Engine (`src/batching.py`)
* **Ticket-Future Decoupling:** Requests register an unfulfilled `asyncio.Future` upon queue entry. The client `await`s its individual future without blocking the primary event loop.
* **Bounded Wait Thresholds:** Collects tensors up to `max_batch_size=32`. If traffic is sparse, it triggers automatically after `max_delay_ms=10.0` ms to eliminate tail latency stalls.
* **Zero Host Contention:** Individual NumPy arrays are stacked along axis 0 in a single vectorized memory allocation immediately before inference execution.

### Quantization & Graph Optimization
* **INT8 Dynamic Quantization:** ResNet architecture quantized from 32-bit floating point to 8-bit integers using ONNX Runtime quantization toolkits.
* **Memory Footprint Reduction:** Memory utilization per worker dropped from ~500 MB to ~145 MB, enabling high-density orchestration on low-spec edge compute.
* **Execution Providers:** Optimized CPU execution using thread affinity pools tuned specifically for 2 dedicated vCPUs.

### Multi-Stage Hardened Containerization
* **Builder-Runner Pattern:** Compilation dependencies (`gcc`, `build-essential`) are restricted strictly to Stage 1 (`builder`).
* **Non-Root Principle of Least Privilege:** Executes under a dedicated system user (`appuser`, UID `10001`) with read-only root system privileges.
* **Kernel Resource Caging:** Verified under explicit Linux cgroups limits: `--cpus="2.0" --memory="1g"`.

---

## 3. Directory Layout

```text
block2_mlops/
├── .dockerignore                  # Strict context boundary filter
├── Dockerfile                     # Multi-stage hardened production blueprint
├── locustfile.py                  # Gevent load test definition (Poisson process)
├── requirements.txt               # Pinned production runtime dependencies
├── models/
│   ├── cifar10_resnet_int8.onnx   # Quantized INT8 ONNX compute graph
│   └── model_manifest.json       # Cryptographic provenance & metadata record
├── scripts/
│   └── benchmark_concurrency.py   # Asynchronous client-side burst validator
├── src/
│   ├── api.py                     # FastAPI application layer & Prometheus hooks
│   ├── batching.py                # Asynchronous queue-based batching engine
│   └── schemas.py                 # Pydantic v2 tensor & image contract validation
└── tests/
    └── test_api.py                # End-to-end API integration and validation tests
```

---

## 4. Empirical Performance & Saturation Benchmarks

Stress-tested using **Locust** in a headless environment targeting the container constrained to **2.0 vCPUs** and **1.0 GB RAM**.

### Sustained Concurrency Test (200 Concurrent Users)

```text
Command: locust -f locustfile.py --headless -u 200 -r 20 --run-time 1m --host http://localhost:8000
```

| Metric Dimension | Value | Operational Evaluation |
| :--- | :--- | :--- |
| **Total Requests Dispatched** | **3,314** | High throughput sustained over 60 seconds |
| **Failures / Drops** | **0 (0.00%)** | Zero dropped connections, socket timeouts, or 5xx errors |
| **Sustained Throughput** | **~55.2 req/s** | Maximum sustained saturation point on 2 vCPUs |
| **Average Batch Size Formed** | **~17.4 - 21.0** | Proves dynamic batching aggregates under peak ingress |
| **Peak Memory Ingest** | **~145.6 MB** | Stable; far below the 1,024 MB kernel cgroup boundary |

### Latency Distribution Profile

```text
Response time percentiles (approximated across 3,314 inference calls):
---------------------------------------------------------------------------------
  50% (Median) :  3,100 ms
  66%          :  3,400 ms
  75%          :  3,500 ms
  80%          :  3,600 ms
  90%          :  3,800 ms
  95%          :  4,000 ms
  99%          :  4,400 ms
 100% (Max)    :  5,300 ms
---------------------------------------------------------------------------------
```

> **Systems Note on Latency:** Under heavy saturation (200 continuous virtual users bombarding 2 CPU cores), median latency rises predictably to ~3.1s due to orderly queue ingress backpressure. The dynamic batcher maintains zero request drops and protects the process from thrashing or running out of memory.

---

## 5. Quickstart & Deployment Guide

### Prerequisites
* Docker Engine 24.0+ or Docker Desktop
* Python 3.12+ (for running client test scripts)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/cifar10-mlops-microservice.git
cd cifar10-mlops-microservice
```

### 2. Build the Hardened Docker Image
```bash
docker build -t cifar10-service:v1 .
```

### 3. Run the Container with Strict Production Constraints
```bash
docker run -d \
  --name cifar10_app \
  -p 8000:8000 \
  --cpus="2.0" \
  --memory="1g" \
  cifar10-service:v1
```

### 4. Verify System Readiness
```bash
# Check service health
curl -i http://localhost:8000/healthz

# Scrape live Prometheus metrics
curl http://localhost:8000/metrics
```

---

## 6. API Specification & Inference Contracts

### `POST /predict`
Processes a single image via base64 encoding or a normalized raw tensor.

#### Sample Request (Normalized Tensor Array)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tensor": [[[0.1, 0.2, ...], [0.3, ...], ...]]
  }'
```

#### Sample Response (`HTTP 200 OK`)
```json
{
  "class_id": 3,
  "class_name": "cat",
  "confidence": 0.8924,
  "inference_time_ms": 14.28,
  "batched_size": 18
}
```

---

## 7. Model Governance & Provenance

The service runtime enforces strict governance standards recorded in `models/model_manifest.json`:
* **Cryptographic Hash (SHA256):** Prevents weight tampering and detects silent file corruption.
* **Input Layout:** Strict Pydantic enforcement of channel-first tensors: `[1, 3, 32, 32]`.
* **Normalization Schema:** $\mu = [0.4914, 0.4822, 0.4465]$, $\sigma = [0.2023, 0.1994, 0.2010]$.

---

## 8. License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.