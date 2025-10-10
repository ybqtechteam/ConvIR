import onnxruntime as ort
import torch
import numpy as np

onnx_model_quantized = "/home/giovannidistasio/vista/models/ConvIR/Dehazing/OTS/convIR_int8.onnx"
session_orig = ort.InferenceSession(onnx_model_quantized, providers=['CPUExecutionProvider'])

onnx_input_name_orig = session_orig.get_inputs()[0].name

dummy_input = torch.randn(1, 3, 480, 640)  # batch=1, 3 channels, H=480, W=640

# Convert PyTorch tensor to NumPy array
dummy_input_np = dummy_input.numpy().astype(np.float32)

# Run inference
onnx_outputs_orig = session_orig.run(None, {onnx_input_name_orig: dummy_input_np})

print("Output shape:", [out.shape for out in onnx_outputs_orig])

