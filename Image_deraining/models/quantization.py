import onnxruntime as ort
import torch 

class ONNXModel(torch.nn.Module):
    def __init__(self, onnx_path):
        super().__init__()
        self.session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().cpu().numpy()
        outputs = self.session.run(None, {self.input_name: x})
        return torch.from_numpy(outputs[2])



if __name__ == "__main__":
    model = ONNXModel("/home/lorenzomignone/gitlab/vista/models/ConvIR/Dehazing/OTS/ckpt/convIR_int8.onnx")
    x = torch.randn(1, 3, 480, 640).to('cuda')
    y = model(x)
    print(y.shape)
