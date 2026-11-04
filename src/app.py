import base64
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Response, status
import onnxruntime as orts
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.schemas import PredictRequest, PredictResponse
from src.batching import AsyncBatcher, preprocess_image_bytes, preprocess_tensor

MODEL_PATH = Path("models/cifar10_resnet_int8.onnx")

REQUEST_COUNT=Counter("inference_request_total","Total requests received",["status"])
INFERENCE_LATENCY=Histogram("inference_latency_seconds","Inference execution latency")

#batcher: AsyncBatcher=None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global batcher
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file missing at: {MODEL_PATH}")
    
    session_opts=orts.SessionOptions()
    session_opts.graph_optimization_level=orts.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.intra_op_num_threads=1
    session_opts.execution_mode=orts.ExecutionMode.ORT_SEQUENTIAL
    
    session=orts.InferenceSession(
        str(MODEL_PATH),
        session_opts=session_opts,
        providers=["CPUExecutionProvider"]
    )
    
    batcher=AsyncBatcher(session=session,max_batch_size=32, max_wait_ms=5.0)
    batcher.start()
    app.state.batcher = batcher
    yield
    await batcher.stop()
    
app=FastAPI(
    title="ResNet-9 CIFAR-10 Edge Inference Service",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/healthz",status_code=status.HTTP_200_OK)
def healthz():
    if batcher is None or batcher.session is None:
        raise HTTPException(status_code=503, detail="Model session is not ready")
    return {"status":"ready", "model":str(MODEL_PATH)}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    batcher = getattr(app.state, "batcher", None)
    if batcher is None:
        raise HTTPException(status_code=503, detail="Model batcher is not initialized")
    
    if not payload.image_b64 and payload.tensor is None:
        REQUEST_COUNT.labels(status="400").inc()
        raise HTTPException(
            status_code=400,
            detail="Payload must include either 'image_b64' or 'tensor'"
        )
    try:
        if payload.image_b64:
            image_raw=base64.b64decode(payload.image_b64)
            preprocessed=preprocess_image_bytes(image_raw)
        else:
            preprocessed=preprocess_tensor(payload.tensor)
            
        with INFERENCE_LATENCY.time():
            result=await batcher.predict(preprocessed)
            
        REQUEST_COUNT.labels(status="200").inc()
        return result
    
    except Exception as exc:
        REQUEST_COUNT.labels(status="500").inc()
        raise HTTPException(status_code=500, detail=str(exc))
    