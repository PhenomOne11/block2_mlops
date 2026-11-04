import argparse
from pathlib import Path
import numpy as np
import onnx
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod
)
from onnxruntime.quantization.shape_inference import quant_pre_process

class CIFAR10CalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_count=100, batch_size=1):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914,0.4822,0.44650],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
        dataset=torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=False,
            transform=transform
        )
        subset=Subset(dataset, range(calibration_count))
        self.loader=DataLoader(subset, batch_size=batch_size, shuffle=False)
        self.iterator=iter(self.loader)
        
    def get_next(self):
        try:
            batch, _ =next(self.iterator)
            return { "input": batch.numpy().astype(np.float32)}
        except StopIteration:
            return None
        
    def rewind(self):
        self.iterator=iter(self.loader)
        
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=str, default="models/cifar10_resnet.onnx")
    parser.add_argument("--output",type=str,default="models/cifar10_resnet_int8.onnx")
    parser.add_argument("--samples", type=int, default=500)
    return parser.parse_args()

def find_sensitive_nodes(onnx_path: Path):
    model = onnx.load(str(onnx_path))
    compute_nodes = [n.name for n in model.graph.node if n.op_type in ["Conv", "Gemm", "MatMul"]]
    exclude = []
    if compute_nodes:
        exclude.append(compute_nodes[0])  
        exclude.append(compute_nodes[-1])  
    print(f"Excluding sensitive boundary nodes from quantization: {exclude}")
    return exclude

def run_static_quantization(input_path: Path, output_path: Path, sample_count: int):
    if not input_path.is_file():
        raise FileNotFoundError(f"Input ONNX model missing at: {input_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    preprocessed_path = input_path.parent / f"{input_path.stem}_preprocessed.onnx"
    quant_pre_process(str(input_path), str(preprocessed_path))
    
    nodes_to_exclude = find_sensitive_nodes(input_path)
    reader=CIFAR10CalibrationDataReader(calibration_count=sample_count)
    
    quantize_static(
        model_input=str(preprocessed_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Entropy,
        per_channel=True,
        nodes_to_exclude=nodes_to_exclude
    )
    
    if preprocessed_path.exists():
        preprocessed_path.unlink()
    
def main():
    args=parse_args()
    in_path=Path(args.input)
    out_path=Path(args.output)
    run_static_quantization(in_path, out_path, args.samples)
    print("SUCCESS")
    
if __name__=="__main__":
    main()