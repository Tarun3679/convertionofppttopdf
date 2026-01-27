# Docling Document Conversion Service - Design Document

**Version:** 2.0 (Verified against official documentation)  
**Date:** January 27, 2026  
**Status:** Draft  
**Author:** Senior Software Engineer

---

## Document Verification Statement

This design document has been verified against the official docling-serve documentation and source references:

- **Official Repository:** https://github.com/docling-project/docling-serve
- **Official Documentation:** https://github.com/docling-project/docling-serve/blob/main/docs/
- **Current Stable Version:** v1.10.0 (v1 API stable since July 14, 2025)
- **License:** MIT

---

## 1. Executive Summary

This document outlines the design for a document conversion microservice using **docling-serve**, IBM's official production-ready API wrapper for the Docling document conversion library. The service will convert various document formats (PDF, DOCX, PPTX, XLSX, HTML, images, audio) into structured outputs (Markdown, JSON, HTML, text, DocTags).

### Key Discovery

The official **docling-serve** project provides a production-ready solution with:
- FastAPI-based REST API with stable v1 endpoints
- Three compute engines: Local, RQ (Redis Queue), KFP (Kubeflow Pipelines)
- Pre-built container images for CPU, CUDA 12.6, CUDA 12.8, and ROCm
- Built-in async processing with WebSocket status updates
- Gradio-based UI for testing

**Recommendation:** Deploy docling-serve directly with minimal customization, using RQ engine for production scalability.

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Web App  │  │ CLI Tool │  │ REST API │  │ Gradio UI│        │
│  │ Consumer │  │          │  │ Consumer │  │ /ui      │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
└───────┼─────────────┼─────────────┼─────────────┼───────────────┘
        │             │             │             │
        └─────────────┼─────────────┼─────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    docling-serve (FastAPI)                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    API Endpoints                         │   │
│  │  /v1/convert/source    /v1/convert/file                 │   │
│  │  /v1/convert/source/async  /v1/convert/file/async       │   │
│  │  /v1/chunk/source      /v1/chunk/file                   │   │
│  │  /v1/status/poll/{id}  /v1/status/ws/{id}               │   │
│  │  /v1/result/{id}       /v1/clear/converters             │   │
│  │  /health               /docs                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌───────────────────────────┼───────────────────────────────┐ │
│  │              Orchestrator (Engine Selection)              │ │
│  │  ┌─────────┐    ┌─────────────┐    ┌──────────────┐      │ │
│  │  │  Local  │    │     RQ      │    │     KFP      │      │ │
│  │  │(Threads)│    │(Redis Queue)│    │  (Kubeflow)  │      │ │
│  │  └────┬────┘    └──────┬──────┘    └──────┬───────┘      │ │
│  └───────┼────────────────┼──────────────────┼───────────────┘ │
│          │                │                  │                  │
│          ▼                ▼                  ▼                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               DocumentConverter (Docling Core)          │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │   │
│  │  │ Layout Model│  │ TableFormer  │  │   OCR Engine  │  │   │
│  │  │  (Heron)    │  │              │  │ (EasyOCR/etc) │  │   │
│  │  └─────────────┘  └──────────────┘  └───────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Compute Engine Options

| Engine | Description | Use Case |
|--------|-------------|----------|
| **Local** | Tasks run in threads within same process | Development, single-instance |
| **RQ** | Redis Queue with separate worker processes | Production, horizontal scaling |
| **KFP** | Kubeflow Pipelines integration | Kubernetes-native ML platforms |

---

## 3. API Specification (v1 - Verified)

### 3.1 Conversion Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/convert/source` | POST | Synchronous conversion from URL/base64 |
| `/v1/convert/source/async` | POST | Async conversion, returns task_id |
| `/v1/convert/file` | POST | Sync conversion from multipart upload |
| `/v1/convert/file/async` | POST | Async file upload conversion |

### 3.2 Status & Result Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/status/poll/{task_id}` | GET | Poll async task status |
| `/v1/status/ws/{task_id}` | WS | WebSocket real-time status |
| `/v1/result/{task_id}` | GET | Retrieve completed task result |

### 3.3 Utility Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chunk/source` | POST | Chunk documents for RAG |
| `/v1/chunk/file` | POST | Chunk uploaded files |
| `/v1/clear/converters` | POST | Clear converter cache |
| `/health` | GET | Health check |
| `/docs` | GET | OpenAPI documentation |
| `/scalar` | GET | Scalar API docs |
| `/ui` | GET | Gradio web UI |

### 3.4 Request Format (v1 API)

**Convert from URL:**
```json
{
  "sources": [
    {"kind": "http", "url": "https://example.com/document.pdf"}
  ],
  "options": {
    "from_formats": ["pdf", "docx", "pptx", "html", "image", "xlsx"],
    "to_formats": ["md", "json", "html", "text", "doctags"],
    "do_ocr": true,
    "force_ocr": false,
    "ocr_engine": "easyocr",
    "ocr_lang": ["en"],
    "pdf_backend": "dlparse_v4",
    "table_mode": "accurate",
    "do_table_structure": true,
    "include_images": true,
    "images_scale": 2.0,
    "image_export_mode": "embedded",
    "abort_on_error": false
  }
}
```

**Convert from base64:**
```json
{
  "sources": [
    {
      "kind": "base64",
      "base64_string": "<base64-encoded-content>",
      "filename": "document.pdf"
    }
  ],
  "options": { ... }
}
```

### 3.5 Response Format

**Synchronous Success:**
```json
{
  "document": {
    "md_content": "# Document Title\n\n...",
    "json_content": { ... },
    "html_content": "<html>...</html>",
    "text_content": "Plain text...",
    "doctags_content": "..."
  },
  "status": "success",
  "processing_time": 12.5,
  "timings": { ... },
  "errors": []
}
```

**Async Task Status:**
```json
{
  "task_id": "8ed9c545-6003-4e8d-8561-aaa219a9445f",
  "task_status": "started",
  "progress": 0.45
}
```

### 3.6 Authentication

When `DOCLING_SERVE_API_KEY` is set, all requests require:
```
X-Api-Key: <secret-key>
```

For WebSocket connections:
```
ws://host:5001/v1/status/ws/{task_id}?api_key=<secret-key>
```

---

## 4. Configuration Reference (Verified)

### 4.1 Server Configuration (Uvicorn)

| Environment Variable | CLI Option | Default | Description |
|---------------------|------------|---------|-------------|
| `UVICORN_HOST` | `--host` | `0.0.0.0` | Host to serve on |
| `UVICORN_PORT` | `--port` | `5001` | Port to serve on |
| `UVICORN_WORKERS` | `--workers` | `1` | Number of worker processes |
| `UVICORN_RELOAD` | `--reload` | `false` | Auto-reload on file changes |
| `UVICORN_TIMEOUT` | `--timeout` | `30` | Server response timeout |
| `UVICORN_PROXY_HEADERS` | N/A | `false` | Enable proxy headers |

### 4.2 Application Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DOCLING_SERVE_ENABLE_UI` | `false` | Enable Gradio UI at /ui |
| `DOCLING_SERVE_ENABLE_VERSIONS` | `false` | Enable /version endpoint |
| `DOCLING_SERVE_ENABLE_REMOTE_SERVICES` | `false` | Allow remote API calls (VLM) |
| `DOCLING_SERVE_ALLOW_EXTERNAL_PLUGINS` | `false` | Allow third-party plugins |
| `DOCLING_SERVE_API_KEY` | (none) | API key for authentication |
| `DOCLING_SERVE_SCRATCH_PATH` | (temp) | Scratch workspace directory |
| `DOCLING_SERVE_ARTIFACTS_PATH` | (cache) | Model artifacts path |

### 4.3 Processing Limits

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DOCLING_SERVE_MAX_DOCUMENT_TIMEOUT` | `604800` (7 days) | Max processing time per document (seconds) |
| `DOCLING_SERVE_MAX_NUM_PAGES` | (none) | Maximum pages per document |
| `DOCLING_SERVE_MAX_FILE_SIZE` | (none) | Maximum file size (bytes) |
| `DOCLING_SERVE_MAX_SYNC_WAIT` | `120` | Sync endpoint timeout (seconds) |
| `DOCLING_SERVE_SYNC_POLL_INTERVAL` | `1.0` | Sync polling interval (seconds) |
| `DOCLING_SERVE_SINGLE_USE_RESULTS` | `false` | Delete results after first access |
| `DOCLING_SERVE_SINGLE_USE_RESULTS_DELAY` | `0` | Delay before deleting results |

### 4.4 Engine Configuration

**Engine Selection:**
| Environment Variable | Value | Description |
|---------------------|-------|-------------|
| `DOCLING_SERVE_ENG_KIND` | `local` | Local threaded engine |
| `DOCLING_SERVE_ENG_KIND` | `rq` | Redis Queue engine |
| `DOCLING_SERVE_ENG_KIND` | `kfp` | Kubeflow Pipelines engine |

**Local Engine:**
| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DOCLING_SERVE_ENG_LOC_NUM_WORKERS` | `1` | Number of worker threads |
| `DOCLING_SERVE_ENG_LOC_SHARE_MODELS` | `true` | Share models across workers |

**RQ Engine:**
| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `RQ_REDIS_URL` | (required) | Redis connection URL |
| `DOCLING_SERVE_ENG_RQ_PREFIX` | `docling-serve` | Result key prefix |
| `DOCLING_SERVE_ENG_RQ_CHANNEL` | `docling-serve-status` | Status channel name |
| `DOCLING_SERVE_ENG_RQ_RESULT_TTL` | `86400` | Result TTL (seconds) |

**KFP Engine:**
| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DOCLING_SERVE_ENG_KFP_ENDPOINT` | (required) | KFP API endpoint |
| `DOCLING_SERVE_ENG_KFP_TOKEN` | (auto) | Authentication token |
| `DOCLING_SERVE_ENG_KFP_CA_CERT` | (auto) | CA certificate path |

### 4.5 Docling Performance Settings

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OMP_NUM_THREADS` | `4` | Torch CPU threads |
| `MKL_NUM_THREADS` | (follows OMP) | MKL threads |
| `DOCLING_DEVICE` | (auto) | Device: `cpu`, `cuda`, `mps`, `cuda:0` |
| `DOCLING_BATCH_SIZE` | (default) | Pages per batch |
| `DOCLING_PERF_TIMINGS` | `false` | Enable detailed timings |

### 4.6 Caching Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DOCLING_SERVE_CACHE_SIZE_OPTIONS` | (default) | LRU cache size for converters |
| `DOCLING_SERVE_PRELOAD_MODELS` | `false` | Preload models at startup |

### 4.7 CORS Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DOCLING_SERVE_CORS_ALLOWED_ORIGINS` | `["*"]` | Allowed origins |
| `DOCLING_SERVE_CORS_ALLOWED_METHODS` | `["*"]` | Allowed methods |
| `DOCLING_SERVE_CORS_ALLOWED_HEADERS` | `["*"]` | Allowed headers |

### 4.8 Telemetry Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `DOCLING_SERVE_ENABLE_METRICS` | `false` | Enable Prometheus metrics |
| `DOCLING_SERVE_ENABLE_TRACING` | `false` | Enable OpenTelemetry tracing |

---

## 5. Container Images (Verified)

### 5.1 Available Images

| Image | Description | Architectures | Size |
|-------|-------------|---------------|------|
| `quay.io/docling-project/docling-serve` | Base (PyPI packages) | amd64, arm64 | 4.4-8.7 GB |
| `quay.io/docling-project/docling-serve-cpu` | CPU-only PyTorch | amd64, arm64 | 4.4 GB |
| `quay.io/docling-project/docling-serve-cu126` | CUDA 12.6 | amd64 | ~10 GB |
| `quay.io/docling-project/docling-serve-cu128` | CUDA 12.8 | amd64 | ~11 GB |
| `docling-serve-rocm` | AMD ROCm 6.3 | amd64 | ~15 GB |

**Note:** ROCm image not published due to size; build locally with `make docling-serve-rocm-image`.

### 5.2 Image Registries

Images available on:
- GitHub Container Registry: `ghcr.io/docling-project/docling-serve*`
- Quay.io: `quay.io/docling-project/docling-serve*`

### 5.3 Model Cache Location

Pre-downloaded models in container: `/opt/app-root/src/.cache/docling/models`

Override with: `DOCLING_SERVE_ARTIFACTS_PATH`

---

## 6. Deployment Patterns

### 6.1 Quick Start (Development)

```bash
# Using Docker
docker run -p 5001:5001 \
  -e DOCLING_SERVE_ENABLE_UI=true \
  quay.io/docling-project/docling-serve

# Verify
curl http://localhost:5001/health
```

### 6.2 Docker Compose - Production (CPU)

```yaml
version: '3.8'

services:
  docling-serve:
    image: quay.io/docling-project/docling-serve-cpu
    container_name: docling-serve
    restart: unless-stopped
    ports:
      - "5001:5001"
    environment:
      - DOCLING_SERVE_ENABLE_UI=true
      - DOCLING_SERVE_MAX_SYNC_WAIT=300
      - DOCLING_SERVE_ENG_LOC_NUM_WORKERS=2
      - OMP_NUM_THREADS=4
      - MKL_NUM_THREADS=4
    volumes:
      - docling-cache:/opt/app-root/src/.cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'

volumes:
  docling-cache:
```

### 6.3 Docker Compose - GPU (CUDA)

```yaml
version: '3.8'

services:
  docling-serve:
    image: quay.io/docling-project/docling-serve-cu128
    container_name: docling-serve-gpu
    restart: unless-stopped
    ports:
      - "5001:5001"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - DOCLING_SERVE_ENABLE_UI=true
      - DOCLING_SERVE_MAX_SYNC_WAIT=600
      - DOCLING_SERVE_MAX_DOCUMENT_TIMEOUT=1200
      - DOCLING_SERVE_ENG_LOC_NUM_WORKERS=2
      - DOCLING_DEVICE=cuda
      - OMP_NUM_THREADS=8
      - MKL_NUM_THREADS=8
    volumes:
      - docling-cache:/opt/app-root/src/.cache
    shm_size: '2gb'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          memory: 16G
          cpus: '8'

volumes:
  docling-cache:
```

### 6.4 Docker Compose - RQ Workers (Production)

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: docling-redis
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"

  docling-api:
    image: quay.io/docling-project/docling-serve-cpu
    container_name: docling-api
    restart: unless-stopped
    ports:
      - "5001:5001"
    environment:
      - DOCLING_SERVE_ENABLE_UI=true
      - DOCLING_SERVE_ENG_KIND=rq
      - RQ_REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/
      - DOCLING_SERVE_API_KEY=${API_KEY}
    depends_on:
      - redis

  docling-worker:
    image: quay.io/docling-project/docling-serve-cpu
    command: docling-serve rq_worker
    restart: unless-stopped
    environment:
      - RQ_REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/
      - OMP_NUM_THREADS=4
    volumes:
      - docling-cache:/opt/app-root/src/.cache
    depends_on:
      - redis
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 8G
          cpus: '4'

volumes:
  redis-data:
  docling-cache:
```

### 6.5 Kubernetes Deployment

```yaml
---
apiVersion: v1
kind: Secret
metadata:
  name: docling-serve-secrets
type: Opaque
stringData:
  API_KEY: "your-secret-api-key"
  REDIS_PASSWORD: "your-redis-password"
  RQ_REDIS_URL: "redis://:your-redis-password@docling-redis:6379/"

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: docling-serve
  labels:
    app: docling-serve
spec:
  replicas: 2
  selector:
    matchLabels:
      app: docling-serve
  template:
    metadata:
      labels:
        app: docling-serve
    spec:
      containers:
      - name: docling-serve
        image: quay.io/docling-project/docling-serve-cpu:latest
        ports:
        - containerPort: 5001
        env:
        - name: DOCLING_SERVE_ENABLE_UI
          value: "true"
        - name: DOCLING_SERVE_ENG_KIND
          value: "rq"
        - name: RQ_REDIS_URL
          valueFrom:
            secretKeyRef:
              name: docling-serve-secrets
              key: RQ_REDIS_URL
        - name: DOCLING_SERVE_API_KEY
          valueFrom:
            secretKeyRef:
              name: docling-serve-secrets
              key: API_KEY
        - name: OMP_NUM_THREADS
          value: "4"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          httpGet:
            path: /health
            port: 5001
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 5001
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: cache
          mountPath: /opt/app-root/src/.cache
      volumes:
      - name: cache
        emptyDir: {}

---
apiVersion: v1
kind: Service
metadata:
  name: docling-serve
spec:
  selector:
    app: docling-serve
  ports:
  - port: 5001
    targetPort: 5001
  type: ClusterIP

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: docling-serve-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: docling-serve
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 7. Client Usage Examples

### 7.1 Synchronous Conversion (curl)

```bash
curl -X 'POST' \
  'http://localhost:5001/v1/convert/source' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'X-Api-Key: your-api-key' \
  -d '{
    "sources": [{"kind": "http", "url": "https://arxiv.org/pdf/2501.17887"}]
  }'
```

### 7.2 Asynchronous Conversion (Python)

```python
import httpx
import time

BASE_URL = "http://localhost:5001"
API_KEY = "your-api-key"
HEADERS = {"X-Api-Key": API_KEY}

# Submit async job
response = httpx.post(
    f"{BASE_URL}/v1/convert/source/async",
    json={
        "sources": [{"kind": "http", "url": "https://arxiv.org/pdf/2501.17887"}],
        "options": {"to_formats": ["md", "json"]}
    },
    headers=HEADERS,
    timeout=60.0
)
task = response.json()
task_id = task["task_id"]
print(f"Task submitted: {task_id}")

# Poll for completion
while True:
    status_response = httpx.get(
        f"{BASE_URL}/v1/status/poll/{task_id}",
        headers=HEADERS
    )
    status = status_response.json()
    print(f"Status: {status['task_status']}")
    
    if status["task_status"] in ("success", "failure"):
        break
    time.sleep(2)

# Retrieve result
if status["task_status"] == "success":
    result = httpx.get(
        f"{BASE_URL}/v1/result/{task_id}",
        headers=HEADERS
    )
    print(result.json()["document"]["md_content"][:500])
```

### 7.3 File Upload (Python)

```python
import httpx
import json

with open("document.pdf", "rb") as f:
    files = {"files": ("document.pdf", f, "application/pdf")}
    data = {
        "parameters": json.dumps({
            "to_formats": ["md", "json"],
            "do_ocr": True,
            "ocr_engine": "easyocr"
        })
    }
    response = httpx.post(
        "http://localhost:5001/v1/convert/file",
        files=files,
        data=data,
        headers={"X-Api-Key": "your-api-key"},
        timeout=300.0
    )
    print(response.json())
```

### 7.4 WebSocket Status Monitoring

```python
import asyncio
import websockets
import json

async def monitor_task(task_id: str, api_key: str):
    uri = f"ws://localhost:5001/v1/status/ws/{task_id}?api_key={api_key}"
    
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            status = json.loads(message)
            print(f"Progress: {status.get('progress', 0):.1%}")
            
            if status["task_status"] in ("success", "failure"):
                break
    
    return status

# Usage
status = asyncio.run(monitor_task("task-id", "api-key"))
```

---

## 8. Supported Formats (Verified)

### 8.1 Input Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| PDF | `.pdf` | Native + OCR support |
| Word | `.docx` | Office Open XML |
| PowerPoint | `.pptx` | Office Open XML |
| Excel | `.xlsx` | Office Open XML |
| HTML | `.html`, `.htm` | Web pages |
| Markdown | `.md` | Pass-through with parsing |
| AsciiDoc | `.adoc` | Documentation format |
| CSV | `.csv` | Tabular data |
| Images | `.png`, `.jpg`, `.jpeg`, `.tiff` | Via OCR |
| Audio | `.wav`, `.mp3` | Via ASR (Whisper) |
| Video Captions | `.vtt` | WebVTT format |
| USPTO XML | `.xml` | Patent documents |
| JATS XML | `.xml` | Journal articles |
| Docling JSON | `.json` | Re-import |

### 8.2 Output Formats

| Format | Description |
|--------|-------------|
| `md` | Markdown (default) |
| `json` | Structured JSON (DoclingDocument) |
| `yaml` | YAML format |
| `html` | HTML document |
| `html_split_page` | HTML with page splits |
| `text` | Plain text |
| `doctags` | Tagged format for ML |

### 8.3 OCR Engines

| Engine | Environment Variable | Notes |
|--------|---------------------|-------|
| `auto` | Default | Auto-select best |
| `easyocr` | Default included | 80+ languages |
| `tesserocr` | Requires tesseract | Fast, accurate |
| `tesseract` | CLI-based | Alternative |
| `rapidocr` | ONNX-based | Fast |
| `ocrmac` | macOS only | Native macOS OCR |

---

## 9. Performance Tuning

### 9.1 Recommended Settings by Workload

**Low Volume (< 100 docs/day):**
```bash
DOCLING_SERVE_ENG_KIND=local
DOCLING_SERVE_ENG_LOC_NUM_WORKERS=2
OMP_NUM_THREADS=4
```

**Medium Volume (100-1000 docs/day):**
```bash
DOCLING_SERVE_ENG_KIND=rq
# 3-5 workers
OMP_NUM_THREADS=4
DOCLING_SERVE_MAX_SYNC_WAIT=300
```

**High Volume (1000+ docs/day):**
```bash
DOCLING_SERVE_ENG_KIND=rq
# 10+ workers with auto-scaling
OMP_NUM_THREADS=8
DOCLING_SERVE_PRELOAD_MODELS=true
# Use GPU images
```

### 9.2 Memory Requirements

| Configuration | Memory per Instance |
|--------------|---------------------|
| API Server | 0.5-1 GB |
| CPU Worker | 4-8 GB |
| GPU Worker | 8-16 GB |

### 9.3 GPU Considerations

- GPU provides 2-5x speedup for OCR-heavy workloads
- VRAM usage: ~4.5 GB after processing (stabilizes)
- Use `DOCLING_DEVICE=cuda:N` for multi-GPU

---

## 10. Troubleshooting

### 10.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| 504 Gateway Timeout | Sync timeout exceeded | Use async endpoints or increase `MAX_SYNC_WAIT` |
| OOM errors | Large documents | Increase container memory limits |
| Slow processing | CPU bound | Increase `OMP_NUM_THREADS`, add workers |
| Model loading slow | Cold start | Set `PRELOAD_MODELS=true` |
| API key rejected | Missing header | Use `X-Api-Key` header |

### 10.2 Debug Mode

```bash
# Enable detailed logging
docker run -p 5001:5001 \
  -e DOCLING_PERF_TIMINGS=true \
  -e LOG_LEVEL=DEBUG \
  quay.io/docling-project/docling-serve
```

### 10.3 Health Monitoring

```bash
# Basic health check
curl http://localhost:5001/health

# Prometheus metrics (if enabled)
curl http://localhost:5001/metrics
```

---

## 11. Security Considerations

1. **API Authentication:** Always set `DOCLING_SERVE_API_KEY` in production
2. **Network:** Use TLS termination at load balancer/ingress
3. **Container:** Run as non-root user (default in official images)
4. **Input Validation:** Service validates file types and sizes
5. **CORS:** Configure appropriate origins in production

---

## 12. References

- **Official Repository:** https://github.com/docling-project/docling-serve
- **Documentation:** https://github.com/docling-project/docling-serve/tree/main/docs
- **Docling Core:** https://github.com/docling-project/docling
- **Technical Report:** https://arxiv.org/abs/2501.17887
- **Container Images:** https://quay.io/organization/docling-project

---

## Appendix A: Migration from v0 to v1

The v1 API (stable since July 14, 2025) introduced breaking changes:

| v0 (deprecated) | v1 (current) |
|-----------------|--------------|
| `/v1alpha/convert/source` | `/v1/convert/source` |
| `http_sources: [{"url": "..."}]` | `sources: [{"kind": "http", "url": "..."}]` |
| `file_sources: [{"base64_string": "..."}]` | `sources: [{"kind": "base64", "base64_string": "...", "filename": "..."}]` |

Key v1 improvements:
- Multiple source types in single request
- Configurable output targets (inline, ZIP, presigned URLs)
- Unified request model via `ConvertDocumentsRequestOptions`
