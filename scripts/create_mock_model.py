import os
import onnx
from onnx import helper, TensorProto

def make_mock_cifar10_onnx(output_path="models/cifar10_resnet_int8.onnx"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Input: [N, 3, 32, 32], Output: [N, 10]
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, 32, 32])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 10])

    flatten_node = helper.make_node("Flatten", inputs=["input"], outputs=["flat"], axis=1)

    starts_t = helper.make_tensor("starts", TensorProto.INT64, [1], [0])
    ends_t = helper.make_tensor("ends", TensorProto.INT64, [1], [10])
    axes_t = helper.make_tensor("axes", TensorProto.INT64, [1], [1])
    slice_node = helper.make_node("Slice", inputs=["flat", "starts", "ends", "axes"], outputs=["output"])

    graph = helper.make_graph(
        [flatten_node, slice_node],
        "mock_cifar10",
        [input_info],
        [output_info],
        initializer=[starts_t, ends_t, axes_t]
    )

    # Force IR version 10 and opset 17 for full ONNX Runtime compatibility
    opset = helper.make_opsetid("", 17)
    model = helper.make_model(graph, producer_name="ci_mock", opset_imports=[opset], ir_version=10)
    
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Created compatible ONNX model (IR=10, opset=17) at {output_path}")

if __name__ == "__main__":
    make_mock_cifar10_onnx()
