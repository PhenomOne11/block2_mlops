import argparse
import gc
import os
import time
from pathlib import Path
import numpy as np
import psutil
import onnxruntime as ort
import torch
import torch.nn as nn

def conv_block(in_channels,out_channels,pool=False):
    layers=[
        nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class Resnet9(nn.Module):
    def __init__(self,in_channels=3,num_classes=10):
        super().__init__()
        self.prep=conv_block(in_channels,64)
        self.layer1_conv=conv_block(64,128,pool=True)
        self.res1=nn.Sequential(conv_block(128,128), conv_block(128,128))
        self.layer2_conv=conv_block(128,256,pool=True)
        self.layer3_conv=conv_block(256,512,pool=True)
        self.res2=nn.Sequential(conv_block(512,512), conv_block(512,512))
        self.pool=nn.AdaptiveMaxPool2d((1,1))
        self.flat = nn.Flatten()
        self.drop=nn.Dropout(0.2)
        self.fc=nn.Linear(512,num_classes)
        
    
    def forward(self,x):
        out=self.prep(x)
        out=self.layer1_conv(out)
        out=out+self.res1(out)
        out=self.layer2_conv(out)
        out=self.layer3_conv(out)
        out=out+self.res2(out)
        out=self.pool(out)
        out=self.flat(out)
        out=self.drop(out)
        out=self.fc(out)
        return out
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_weights", type=str, default="models/cifar10_resnet_91.85.pt")
    parser.add_argument("--onnx_fp32", type=str, default="models/cifar10_resnet.onnx")
    parser.add_argument("--onnx_int8", type=str, default="models/cifar10_resnet_int8.onnx")
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    return parser.parse_args()

def get_current_rss():
    process=psutil.Process(os.getpid())
    return process.memory_info().rss/(1024*1024)

def benchmark_pytorch(weights_path: Path, runs:int, warmup:int):
    gc.collect()
    device=torch.device("cpu")
    model=Resnet9(in_channels=3,num_classes=10)
    state_dict=torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    torch.set_num_threads(1)
    
    sample=torch.randn(1,3,32,32, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for _ in range(warmup):
            _=model(sample)
            
        latencies=[]
        for _ in range(runs):
            t0=time.perf_counter()
            _=model(sample)
            t1=time.perf_counter()
            latencies.append((t1-t0)*1000.0)
            
    rss=get_current_rss()
    disk_size=weights_path.stat().st_size/(1024*1024)
    
    del model
    del state_dict
    gc.collect()
    
    return {
        "disk_mb": disk_size,
        "rss_mb": rss,
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99)
    }
    
def benchmark_onnx(model_path: Path, runs:int, warmup: int):
    gc.collect()
    session_options=ort.SessionOptions()
    session_options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    session=ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"]
    )
    
    input_name=session.get_inputs()[0].name
    output_name=session.get_outputs()[0].name
    sample = np.random.randn(1, 3, 32, 32).astype(np.float32)
    
    for _ in range(warmup):
        _=session.run([output_name], {input_name: sample})
        
    latencies=[]
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = session.run([output_name], {input_name: sample})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    rss = get_current_rss()
    disk_size = model_path.stat().st_size / (1024 * 1024)

    del session
    gc.collect()

    return {
        "disk_mb": disk_size,
        "rss_mb": rss,
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99)
    }
    
def display_results(results: dict):
    header = f"{'Runtime':<24} | {'Size (MB)':<10} | {'RSS (MB)':<10} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'p99 (ms)':<10}"
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)
    for name, m in results.items():
        print(f"{name:<24} | {m['disk_mb']:<10.2f} | {m['rss_mb']:<10.2f} | {m['p50']:<10.3f} | {m['p95']:<10.3f} | {m['p99']:<10.3f}")
    print(separator + "\n")
    
def main():
    args=parse_args()
    
    results={}
    print("[1/3] Benchmarking Pytorch Eager(FP32)....")
    results["Pytorch Eager (FP32)"]=benchmark_pytorch(
        Path(args.pytorch_weights), args.runs, args.warmup
    )
    
    print("[2/3] Benchmarking ONNX Runtime (FP32)...")
    results["ONNX Runtime (FP32)"]=benchmark_onnx(
        Path(args.onnx_fp32), args.runs, args.warmup
    )
    
    print("[3/3] Benchmarking ONNX Runtime (INT8)...")
    results["ONNX Runtime (INT8)"] = benchmark_onnx(
        Path(args.onnx_int8), args.runs, args.warmup
    )

    display_results(results)
    
if __name__ == "__main__":
    main()