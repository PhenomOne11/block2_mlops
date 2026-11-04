import argparse
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.prep = conv_block(in_channels, 64)
        self.layer1_conv = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.layer2_conv = conv_block(128, 256, pool=True)
        self.layer3_conv = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.prep(x)
        out = self.layer1_conv(out)
        out = out + self.res1(out)
        out = self.layer2_conv(out)
        out = self.layer3_conv(out)
        out = out + self.res2(out)
        out = self.pool(out)
        out = self.flat(out)
        out = self.drop(out)
        out = self.fc(out)
        return out
    
    
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/cifar10_resnet.onnx")
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()

def load_model(weights_path:str,device: torch.device) -> nn.Module:
    model=ResNet9(num_classes=10)
    checkpoint=Path(weights_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing weights file: {checkpoint}")
    
    state_dict=torch.load(checkpoint,map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def export_graph(model: nn.Module, output_path: Path, opset_version: int, device: torch.device):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input=torch.randn(1,3,32,32, device=device, dtype=torch.float32)
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
    with torch.no_grad():
        torch.onnx.export(
            model=model,
            args=(dummy_input),
            f=str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            dynamo=False
        )
        
def verify_structure(output_path: Path):
    model_proto=onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)

def audit_numerical_parity(model: nn.Module, output_path: Path, device: torch.device):
    session= ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    for batch_size in [1,4,16]:
        sample=torch.randn(batch_size, 3,32,32,device=device)
        with torch.no_grad():
            expected=model(sample).cpu().numpy()
    actual = session.run(["output"], {"input": sample.cpu().numpy()})[0]
    np.testing.assert_allclose(expected,actual, rtol=1e-3, atol=1e-5)
    
def main():
    args=parse_args()
    device=torch.device("cpu")
    model=load_model(args.weights, device)
    out_path=Path(args.output)
    export_graph(model, out_path, args.opset, device)
    verify_structure(out_path)
    audit_numerical_parity(model,out_path,device)
    print("SUCCESS")
    
if __name__=="__main__":
    main()