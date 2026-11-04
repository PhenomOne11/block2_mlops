from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest


ONNX_PATH = Path("models/cifar10_resnet.onnx")

def test_onnx_file_exists():
    assert ONNX_PATH.is_file(), f"ONNX model file not found at: {ONNX_PATH}"
    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    assert size_mb > 20.0, f"ONNX file size suspiciously small: {size_mb:.2f} MB"
    
def test_onnx_spec_validity():
    model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model)

def test_onnx_dynamic_batching():
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    assert input_meta.name == "input"
    assert output_meta.name == "output"

    for batch_size in [1, 7, 32]:
        dummy_tensor = np.random.randn(batch_size, 3, 32, 32).astype(np.float32)
        outputs = session.run(["output"], {"input": dummy_tensor})[0]
        assert outputs.shape == (batch_size, 10), (
            f"Expected output shape ({batch_size}, 10), got {outputs.shape}"
        )
        assert not np.isnan(outputs).any(), "NaN values detected in model output"
        assert not np.isinf(outputs).any(), "Inf values detected in model output"