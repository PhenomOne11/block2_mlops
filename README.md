# block2_mlops

> **Production-grade, INT8-quantized edge serving microservice featuring custom async dynamic micro-batching. Handles 200 concurrent VUs with 0.00% failure rate at ~55.2 req/s under strict 1 GB RAM / 2-core cgroup resource constraints.**

[![CI](https://github.com/PhenomOne11/block2_mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/PhenomOne11/block2_mlops/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-INT8%20Quantized-emerald.svg)](https://onnxruntime.ai/)

---
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111.0-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/ONNX_Runtime-INT8-005CED?style=for-the-badge&logo=onnx&logoColor=white" alt="ONNX Runtime"/>
  <img src="https://img.shields.io/badge/Docker-Multi--Stage-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/Locust-Load_Testing-00A86B?style=for-the-badge&logo=locust&logoColor=white" alt="Locust"/>
</p>

<p align="center">
  <a href="https://github.com/PhenomOne11/block2_mlops/actions/workflows/ci.yml">
    <img src="https://github.com/PhenomOne11/block2_mlops/actions/workflows/ci.yml/badge.svg" alt="CI Pipeline"/>
  </a>
</p>

**block2_mlops** is an end-to-end machine learning model-serving system designed to deploy an optimized CIFAR-10 ResNet-9 classifier through a containerized inference microservice.

The system combines ONNX Runtime INT8 quantization, asynchronous request ingestion, dynamic micro-batching, FastAPI inference serving, Prometheus observability, Docker containerization, and Locust-based load testing.

Built with a focus on inference efficiency, resource utilization, concurrency management, and deployment reliability.

---

## 1. Model Development & Deployment Pipeline

This repository is the deployment counterpart of **[nanoengine](https://github.com/PhenomOne11/nanoengine)**, the custom deep learning training framework where the ResNet-9 model is developed and trained.

The trained model weights exported from `nanoengine` are converted into an optimized ONNX model and deployed through `block2_mlops`.

| Repository                                              | Primary Responsibility                                                                                                              |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| [nanoengine](https://github.com/PhenomOne11/nanoengine) | Custom ResNet-9 architecture, neural network layers, optimization algorithms, model training and weight export                      |
| **block2_mlops**                                        | ONNX export and quantization workflow, inference serving, dynamic batching, containerization, monitoring and performance evaluation |

**End-to-end workflow:**

```text
                  MODEL DEVELOPMENT
                         |
                         v
              +----------------------+
              |      nanoengine      |
              |                      |
              |  Custom ResNet-9     |
              |  Training Pipeline   |
              |  Optimizer & Layers  |
              +----------------------+
                         |
                         v
                 Trained Model Weights
                         |
                         v
                  MODEL OPTIMIZATION
                         |
                         v
              +----------------------+
              |     ONNX Runtime     |
              |                      |
              |  ONNX Model Export   |
              |  INT8 Quantization   |
              +----------------------+
                         |
                         v
                 DEPLOYMENT PIPELINE
                         |
                         v
              +----------------------+
              |     block2_mlops     |
              |                      |
              |  FastAPI Serving     |
              |  Dynamic Batching    |
              |  Docker Deployment   |
              |  Prometheus Metrics  |
              +----------------------+
                         |
                         v
                 PERFORMANCE TESTING
                         |
                         v
              +----------------------+
              |       Locust         |
              |                      |
              |  Load Testing        |
              |  Concurrency Tests   |
              |  Latency Analysis    |
              +----------------------+
```

---

## 2. System Architecture & Request Processing

Traditional request-by-request inference can introduce unnecessary execution overhead when multiple clients send requests concurrently.

This system implements an asynchronous producer-consumer architecture with a queue-based dynamic batching engine.

### Inference Architecture

```text
              +-------------------------+
              |     Concurrent Clients  |
              +-------------------------+
                           |
                           v
              +-------------------------+
              |      FastAPI Server     |
              |      Uvicorn Event Loop |
              +-------------------------+
                           |
                           v
              +-------------------------+
              |   Input Validation      |
              |                         |
              |  Base64 Image / Tensor  |
              |  Pydantic Schemas        |
              |  Preprocessing           |
              +-------------------------+
                           |
                           v
              +-------------------------+
              |    Asyncio Request Queue|
              |                         |
              |  Request + Future       |
              +-------------------------+
                           |
                           v
       +-------------------------------------------+
       |           ASYNCHRONOUS BATCHER             |
       |                                           |
       |  Maximum Batch Size: 32                   |
       |  Maximum Waiting Time: 10 ms              |
       |                                           |
       |  Collects incoming requests               |
       |  Groups tensors into a single batch       |
       |  Stacks inputs into (N, 3, 32, 32)        |
       +-------------------------------------------+
                           |
                           v
              +-------------------------+
              |     ONNX Runtime        |
              |                         |
              |  INT8 Quantized Model   |
              |  CPU Inference          |
              +-------------------------+
                           |
                           v
              +-------------------------+
              |    Result Dispatcher    |
              |                         |
              |  Softmax Probabilities |
              |  Top Predictions       |
              |  Individual Futures    |
              +-------------------------+
                           |
                           v
              +-------------------------+
              |     HTTP Responses      |
              |                         |
              |  Individual Clients     |
              +-------------------------+
```

### Request Lifecycle

1. Multiple clients send prediction requests to the FastAPI endpoint.
2. Incoming payloads are validated and converted into model-compatible tensors.
3. Each request is placed into an asynchronous queue with its corresponding `asyncio.Future`.
4. The batching engine collects requests until the maximum batch size or waiting-time threshold is reached.
5. Individual tensors are combined into a single batch.
6. ONNX Runtime executes inference on the quantized model.
7. Predictions are distributed to the corresponding request futures.
8. Each client receives its individual HTTP response.

The batching mechanism reduces the overhead associated with processing individual requests separately.

---

## 3. Key Engineering Components

### 3.1 Asynchronous Dynamic Micro-Batching

**Implementation:** `src/batching.py`

The batching engine coordinates incoming requests and combines them into larger inference batches.

Key characteristics:

* **Future-Based Request Handling:** Each incoming request receives an individual `asyncio.Future`, allowing its coroutine to await the corresponding prediction result.
* **Bounded Batch Collection:** Collects requests up to a configurable maximum batch size of 32.
* **Timeout-Based Execution:** Processes partially filled batches after the configured waiting threshold of 10 ms.
* **Vectorized Input Preparation:** Combines individual image tensors along the batch dimension.
* **Individual Result Dispatch:** Maps each prediction back to its original request.

**Input transformation:**

```text
Individual Requests:

Request 1 -> (3, 32, 32)
Request 2 -> (3, 32, 32)
Request 3 -> (3, 32, 32)
       ...
Request N -> (3, 32, 32)

             |
             v

Combined Batch:

(N, 3, 32, 32)
```

### 3.2 ONNX Export & INT8 Quantization

The trained ResNet-9 model is deployed using the ONNX Runtime inference engine.

Key components:

* **ONNX Export:** Converts the trained PyTorch model into the ONNX computational graph format.
* **INT8 Quantization:** Uses 8-bit integer quantization to reduce model representation and improve inference efficiency.
* **Calibration:** Uses representative input data to determine quantization parameters for static quantization.
* **CPU Execution:** Uses ONNX Runtime's optimized CPU execution capabilities.

The optimization pipeline aims to reduce inference overhead and improve deployment efficiency on resource-constrained hardware.

### 3.3 FastAPI Inference Service

**Implementation:** `src/api.py`

The FastAPI application exposes HTTP endpoints for model inference and service observability.

Features:

* Asynchronous request handling.
* Pydantic-based input validation.
* Base64 image and normalized tensor input support.
* Dynamic batching integration.
* JSON prediction responses.
* Prometheus metrics integration.

### 3.4 Containerization & Runtime Security

**Implementation:** `Dockerfile`

The application uses a multi-stage Docker build to separate dependency preparation from runtime execution.

**Builder stage:**

* Installs build dependencies.
* Creates an isolated Python virtual environment.
* Installs application dependencies.

**Runtime stage:**

* Uses a clean Python slim image.
* Copies the prepared virtual environment.
* Copies application source code and model artifacts.
* Executes the service using a dedicated non-root user.

Additional runtime configurations include:

* Dedicated application user (`appuser`).
* Explicit CPU and memory constraints.
* Docker health checks.
* Reduced runtime image dependencies.

### 3.5 Monitoring & Observability

The service exposes Prometheus-compatible metrics through `/metrics`.

Monitoring capabilities include:

* Request counts.
* Request latency distribution.
* Inference-related performance measurements.
* Service health checks.

The `/healthz` endpoint provides a basic service readiness check.

### 3.6 Load & Performance Testing

**Implementation:** `locustfile.py`

Locust is used to simulate concurrent clients and evaluate service performance under different workloads.

Evaluation dimensions include:

* Request throughput.
* Response latency percentiles.
* Failure rates.
* Batch sizes.
* Resource consumption.

---

## 4. Project Structure

```text
block2_mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── models/
│   ├── cifar10_resnet_int8.onnx
│   └── model_manifest.json
│
├── scripts/
│   └── benchmark_concurrency.py
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── batching.py
│   └── schemas.py
│
├── tests/
│   └── test_api.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── locustfile.py
├── requirements.txt
└── README.md
```

---

## 5. Empirical Performance & Load Testing

The service was evaluated using Locust in headless mode under explicit Docker resource constraints.

### Test Environment

| Configuration     | Value             |
| ----------------- | ----------------- |
| Load Testing Tool | Locust            |
| Virtual Users     | 200               |
| User Spawn Rate   | 20 users/second   |
| Test Duration     | 60 seconds        |
| CPU Limit         | 2.0 CPUs          |
| Memory Limit      | 1 GB              |
| Model Runtime     | ONNX Runtime INT8 |

### Load Test Command

```bash
locust -f locustfile.py --headless -u 200 -r 20 --run-time 1m --host http://localhost:8000
```

### Sustained Concurrency Results

| Metric               |           Recorded Result |
| -------------------- | ------------------------: |
| Total Requests       |                 **3,314** |
| Failed Requests      |                     **0** |
| Failure Rate         |                 **0.00%** |
| Average Throughput   | **~55.2 requests/second** |
| Average Batch Size   |            **~17.4–21.0** |
| Peak Observed Memory |             **~145.6 MB** |

*Throughput is calculated from the recorded request count divided by the 60-second test duration.*

### Response Latency Distribution

| Percentile | Response Time |
| ---------- | ------------: |
| p50        |      3,100 ms |
| p66        |      3,400 ms |
| p75        |      3,500 ms |
| p80        |      3,600 ms |
| p90        |      3,800 ms |
| p95        |      4,000 ms |
| p99        |      4,400 ms |
| Maximum    |      5,300 ms |

### Performance Interpretation

The service sustained approximately 55 requests per second during the recorded 200-user test, with no reported request failures.

However, the latency distribution shows that response times increased substantially under sustained concurrency.

This indicates that although dynamic batching successfully aggregates incoming requests, queueing delay becomes a significant component of end-to-end latency under heavy load.

The results demonstrate the importance of evaluating throughput, latency, batching efficiency, and resource consumption together rather than treating throughput as the only measure of inference performance.

*Benchmark results are specific to the recorded hardware, workload, Docker constraints, and test configuration. They should not be interpreted as universal performance guarantees.*

---

## 6. Quickstart & Deployment

### Prerequisites

* Docker Engine 24.0+ or Docker Desktop.
* Python 3.12+ (for running local benchmark scripts).
* Git.

### Step 1. Clone the Repository

```bash
git clone https://github.com/PhenomOne11/block2_mlops.git
cd block2_mlops
```

### Step 2. Build the Docker Image

```bash
docker build -t cifar10-service:v1 .
```

### Step 3. Run with Resource Constraints

```bash
docker run -d \
  --name cifar10_app \
  -p 8000:8000 \
  --cpus="2.0" \
  --memory="1g" \
  cifar10-service:v1
```

The container is configured with:

* Maximum CPU allocation of approximately 2 CPU cores.
* Maximum memory allocation of 1 GB.
* Application execution under a non-root user.

### Step 4. Verify Service Health

```bash
curl -i http://localhost:8000/healthz
```

### Step 5. Access Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

### Step 6. Open API Documentation

Visit:

```text
http://localhost:8000/docs
```

FastAPI automatically generates interactive API documentation.

### Useful Docker Commands

Check running containers:

```bash
docker ps
```

View application logs:

```bash
docker logs -f cifar10_app
```

Stop the container:

```bash
docker stop cifar10_app
```

Remove the container:

```bash
docker rm cifar10_app
```

---

## 7. API Specification

### `POST /predict`

Accepts an image encoded as Base64 or a normalized numerical tensor and returns the model's classification result.

**Supported input formats:**

* Base64-encoded image.
* Normalized tensor with shape `(3, 32, 32)`.

### Example Request

The following is a simplified illustration of a normalized tensor payload:

```json
{
  "tensor": "3-channel normalized image tensor of shape (3, 32, 32)"
}
```

*The actual API expects numerical tensor data rather than the explanatory string shown above.*

### Example Response

```json
{
  "class_id": 3,
  "class_name": "cat",
  "confidence": 0.8924,
  "inference_time_ms": 14.28,
  "batched_size": 18
}
```

| Field               | Description                                         |
| ------------------- | --------------------------------------------------- |
| `class_id`          | Predicted CIFAR-10 class index                      |
| `class_name`        | Predicted class label                               |
| `confidence`        | Softmax confidence score                            |
| `inference_time_ms` | Reported inference timing                           |
| `batched_size`      | Number of requests processed in the inference batch |

### Additional Endpoints

| Endpoint   | Method | Purpose              |
| ---------- | ------ | -------------------- |
| `/predict` | POST   | Image classification |
| `/healthz` | GET    | Service health check |
| `/metrics` | GET    | Prometheus metrics   |

---

## 8. Model Governance & Provenance

The deployment includes a model manifest at:

```text
models/model_manifest.json
```

The manifest records model metadata and provenance information.

Key considerations:

* **SHA-256 Integrity Verification:** Cryptographic hashing can be used to detect model file modifications or corruption.
* **Input Contract:** Channel-first image tensors with dimensions `(3, 32, 32)`.
* **Normalization:** CIFAR-10 channel-wise normalization.

Normalization parameters:

```python
mean = [0.4914, 0.4822, 0.4465]
std  = [0.2023, 0.1994, 0.2010]
```

Maintaining consistent preprocessing between training and inference is essential for reliable model predictions.

---

## 9. Testing & Continuous Integration

The repository includes automated API tests and a GitHub Actions CI workflow.

### Run Tests Locally

```bash
pip install -r requirements.txt
pytest tests/ -v
```

### Continuous Integration

The GitHub Actions workflow is configured in:

```text
.github/workflows/ci.yml
```

The CI pipeline provides automated validation of the application through the configured test workflow.

**CI Status:**

[View GitHub Actions](https://github.com/PhenomOne11/block2_mlops/actions)

---

## 10. Technology Stack

| Technology     | Role                                        |
| -------------- | ------------------------------------------- |
| Python 3.12    | Core application language                   |
| PyTorch        | Model development and training ecosystem    |
| ONNX           | Model interchange format                    |
| ONNX Runtime   | Optimized model inference                   |
| FastAPI        | HTTP inference service                      |
| Pydantic       | Request validation                          |
| NumPy          | Numerical operations and tensor preparation |
| asyncio        | Asynchronous request coordination           |
| Docker         | Application containerization                |
| Prometheus     | Metrics collection                          |
| Locust         | Load and concurrency testing                |
| Pytest         | Automated testing                           |
| GitHub Actions | Continuous integration                      |

---

## 11. Engineering Objectives

This project explores practical engineering challenges involved in taking a trained machine learning model from development to deployment.

Its primary objectives include:

* Converting trained PyTorch models into portable ONNX artifacts.
* Applying INT8 quantization for optimized inference.
* Building asynchronous inference request pipelines.
* Implementing dynamic batching for concurrent workloads.
* Deploying models through resource-constrained Docker containers.
* Exposing service-level metrics for observability.
* Evaluating throughput, latency, and resource utilization.
* Establishing automated testing and CI workflows.

---

## 12. License

Distributed under the Apache License 2.0. See the `LICENSE` file for details.

---

<p align="center">
  <b>Part of the nanoengine → block2_mlops machine learning lifecycle.</b>
  <br/>
  <a href="https://github.com/PhenomOne11/nanoengine">Model Development & Training</a>
  &nbsp; | &nbsp;
  <a href="https://github.com/PhenomOne11/block2_mlops">Model Deployment & Inference</a>
</p>
