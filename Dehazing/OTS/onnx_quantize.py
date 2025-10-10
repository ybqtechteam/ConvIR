# from onnxruntime.quantization import quantize_dynamic, QuantType

# onnx_model = "/home/giovannidistasio/vista/models/ConvIR/Dehazing/OTS/convIR.onnx"
# onnx_model_quantized = "/home/giovannidistasio/vista/models/ConvIR/Dehazing/OTS/convIR_int8.onnx"

# quantize_dynamic(
#     model_input=onnx_model,
#     model_output=onnx_model_quantized,
#     weight_type=QuantType.QInt8  # può anche essere QUInt8
# )

# print("Quantizzazione dinamica completata!")







#----------------------------DINAMICA CON CONVERSIONE A FLOAT32--------------------------------


# import onnx
# from onnxruntime.quantization import quantize_dynamic, QuantType
# import numpy as np
# from onnx import numpy_helper, shape_inference

# # Percorsi
# onnx_model_path = "/home/giovannidistasio/vista/models/ConvIR/Dehazing/OTS/convIR.onnx"
# onnx_model_float32_path = onnx_model_path.replace(".onnx", "_float32.onnx")
# onnx_model_quantized_path = onnx_model_path.replace(".onnx", "_int8_dynamic.onnx")

# # 1️⃣ Carica modello ONNX
# model = onnx.load(onnx_model_path)

# # 2️⃣ Converti tutti gli initializer da float64 a float32
# count_init = 0
# for tensor in model.graph.initializer:
#     if tensor.data_type == onnx.TensorProto.DOUBLE:
#         array = np.frombuffer(tensor.raw_data, dtype=np.float64)
#         tensor.raw_data = array.astype(np.float32).tobytes()
#         tensor.data_type = onnx.TensorProto.FLOAT
#         count_init += 1
# print(f"[INFO] Tensori initializer convertiti da float64 a float32: {count_init}")

# # 3️⃣ Converti anche i nodi intermedi con attributi float64 (es. costanti)
# count_const = 0
# for node in model.graph.node:
#     for attr in node.attribute:
#         if attr.type == onnx.AttributeProto.TENSOR:
#             tensor = attr.t
#             if tensor.data_type == onnx.TensorProto.DOUBLE:
#                 array = numpy_helper.to_array(tensor)
#                 tensor.CopyFrom(numpy_helper.from_array(array.astype(np.float32), tensor.name))
#                 count_const += 1
# print(f"[INFO] Tensori costanti nei nodi convertiti da float64 a float32: {count_const}")

# # 4️⃣ Shape inference (aggiorna tipi e dimensioni tensori intermedi)
# model = shape_inference.infer_shapes(model)
# print("[INFO] Shape inference completata.")

# # 5️⃣ Salva modello float32
# onnx.save(model, onnx_model_float32_path)
# print(f"[INFO] Modello float32 salvato in: {onnx_model_float32_path}")

# # 6️⃣ Quantizzazione dinamica INT8 (solo pesi)
# quantize_dynamic(
#     model_input=onnx_model_float32_path,
#     model_output=onnx_model_quantized_path,
#     weight_type=QuantType.QInt8
# )
# print(f"[INFO] Quantizzazione dinamica INT8 completata. Modello salvato in: {onnx_model_quantized_path}")








#--------------------------------------STATICA-----------------------------


import os
import torch
import torch.nn.functional as nnF
import numpy as np
import inspect
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import onnx
from torchvision import transforms
from data.data_load import CalibrationDataset


# === 🔍 Controllo di sicurezza: assicuriamoci che nnF.pad sia quello corretto ===
print("Pad in uso da:", inspect.getmodule(nnF.pad))

# === 1️⃣ Imposta parametri ===
image_dir = "dataset/calibration/hazy/"  # directory con immagini di dehazing
onnx_model = "convIR.onnx"
onnx_model_quantized = "convIR_int8_III.onnx"
batch_size = 1
num_workers = 0
device = "cpu"  # quantizzazione statica usa solo CPU
factor = 8  # come nel tuo codice

# === 2️⃣ Carica nome input modello ===
model = onnx.load(onnx_model)
input_name = model.graph.input[0].name
print(f"Input del modello: {input_name}")

# === 3️⃣ Stesso preprocessing del training ===
transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
])

dataset = CalibrationDataset(image_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False
)

# === 4️⃣ Converte batch di immagini PyTorch in NumPy per ONNX ===
def prepare_image_tensor(input_img, factor=8):
    """
    Applica padding riflessivo come nel training e converte in NumPy float32
    """
    h, w = input_img.shape[2], input_img.shape[3]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_img = nnF.pad(input_img, (0, padw, 0, padh), mode='reflect')
    return input_img.cpu().numpy().astype(np.float32)

# === 5️⃣ Crea un CalibrationDataReader ===
class TorchImageDataReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name, factor):
        self.iterator = iter(dataloader)
        self.input_name = input_name
        self.factor = factor

    def get_next(self):
        try:
            # dataset restituisce solo immagini
            input_img = next(self.iterator)  # non unpacking
            np_input = prepare_image_tensor(input_img, self.factor)
            return {self.input_name: np_input}
        except StopIteration:
            return None

calibration_data_reader = TorchImageDataReader(dataloader, input_name, factor)

# === 6️⃣ Esegui quantizzazione ===
print("Inizio quantizzazione statica...")
quantize_static(
    model_input=onnx_model,
    model_output=onnx_model_quantized,
    calibration_data_reader=calibration_data_reader,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8
)
print(f"✅ Modello quantizzato salvato in: {onnx_model_quantized}")
