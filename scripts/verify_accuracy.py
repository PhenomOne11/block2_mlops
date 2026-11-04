import argparse
from pathlib import Path
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32_model", type=str, default="models/cifar10_resnet.onnx")
    parser.add_argument("--int8_model", type=str, default="models/cifar10_resnet_int8.onnx")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--test_samples", type=int, default=1000)
    return parser.parse_args()

def get_test_loader(batch_size: int, total_samples: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=transform
    )
    indices = list(range(min(total_samples, len(dataset))))
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False)

def evaluate_onnx(model_path: Path, loader: DataLoader):
    session=ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name=session.get_inputs()[0].name
    output_name=session.get_outputs()[0].name
    
    correct=0
    total=0
    
    for images, labels in loader:
        numpy_images=images.numpy().astype(np.float32)
        numpy_labels=labels.numpy()
        
        outputs=session.run([output_name], {input_name: numpy_images})[0]
        predictions=np.argmax(outputs, axis=1)
        
        correct+=np.sum(predictions==numpy_labels)
        total+=labels.size(0)
        
    accuracy=(correct/total)*100.0
    return accuracy

def main():
    import torch
    args = parse_args()

    fp32_path = Path(args.fp32_model)
    int8_path = Path(args.int8_model)

    if not fp32_path.is_file() or not int8_path.is_file():
        raise FileNotFoundError("One or both ONNX model files are missing.")

    print(f"Loading {args.test_samples} CIFAR-10 test samples...")
    test_loader = get_test_loader(args.batch_size, args.test_samples)

    print("Evaluating ONNX FP32 Top-1 Accuracy...")
    fp32_acc = evaluate_onnx(fp32_path, test_loader)
    print(f"FP32 Top-1 Accuracy: {fp32_acc:.2f}%")

    print("Evaluating ONNX INT8 Top-1 Accuracy...")
    int8_acc = evaluate_onnx(int8_path, test_loader)
    print(f"INT8 Top-1 Accuracy: {int8_acc:.2f}%")

    drop = fp32_acc - int8_acc
    print("\n" + "=" * 45)
    print(f"Accuracy Drop: {drop:.2f}%")
    if drop < 1.0:
        print("VERIFICATION GATE: PASSED (< 1.0% drop)")
    else:
        print("VERIFICATION GATE: FAILED (>= 1.0% drop)")
    print("=" * 45)


if __name__ == "__main__":
    main()