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


#--------------------------------------STATICA (MULTI-DIRECTORY)-----------------------------

import os
import torch
import torch.nn.functional as nnF
import numpy as np
import inspect
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, CalibrationMethod
import onnx
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# === 🔍 Controllo di sicurezza: assicuriamoci che nnF.pad sia quello corretto ===
print("Pad in uso da:", inspect.getmodule(nnF.pad))

# === 1️⃣ Imposta parametri ===
image_dirs = [
    "/home/giovannidistasio/vista/models/ConvIR/Dehazing/OTS/dataset/benchmark_splitted/Dense_Haze/train/hazy",
    "/home/giovannidistasio/vista/models/ConvIR/Dehazing/OTS/dataset/benchmark_splitted/Dense_Haze/val/hazy",
    #"/home/giovannidistasio/vista/models/ConvIR/Dehazing/OTS/dataset/benchmark_splitted/Dense_Haze/test/hazy",
    #  "/home/giovannidistasio/vista/models/ConvIR/Dehazing/OTS/dataset/benchmark_splitted/NH-HAZE/test/hazy",
    #  "/home/giovannidistasio/vista/models/ConvIR/Dehazing/OTS/dataset/benchmark_splitted/NH-HAZE/val/hazy",
    #  "/home/giovannidistasio/vista/models/ConvIR/Dehazing/OTS/dataset/benchmark_splitted/NH-HAZE/train/hazy",
    #"/home/giovannidistasio/vista/models/ConvIR/Dehazing/OTS/dataset/custom_dataset_splitted/test/hazy"
]
onnx_model = "convIR.onnx"
onnx_model_quantized = "convIR_int8_excluded.onnx"
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

# === 4️⃣ Dataset personalizzato che supporta più directory ===
class CalibrationDataset(Dataset):
    def __init__(self, image_dirs, transform=None):
        if isinstance(image_dirs, str):
            image_dirs = [image_dirs]
        self.image_paths = []
        for d in image_dirs:
            if not os.path.exists(d):
                print(f"[ATTENZIONE] Cartella non trovata: {d}")
                continue
            files = [
                os.path.join(d, f)
                for f in os.listdir(d)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            self.image_paths.extend(files)

        if not self.image_paths:
            raise RuntimeError("❌ Nessuna immagine trovata nelle cartelle di calibrazione!")
        print(f"[INFO] Trovate {len(self.image_paths)} immagini totali di calibrazione.")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


# === 5️⃣ DataLoader ===
dataset = CalibrationDataset(image_dirs, transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False
)

# === 6️⃣ Prepara immagini per ONNX ===
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

# === 7️⃣ DataReader per la calibrazione ===
class TorchImageDataReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name, factor):
        self.iterator = iter(dataloader)
        self.input_name = input_name
        self.factor = factor

    def get_next(self):
        try:
            input_img = next(self.iterator)
            np_input = prepare_image_tensor(input_img, self.factor)
            return {self.input_name: np_input}
        except StopIteration:
            return None

calibration_data_reader = TorchImageDataReader(dataloader, input_name, factor)


# === 🔎 Filtra i nodi da escludere in base ai prefissi ===
exclude_prefixes = ["/Decoder.2/layers/", "/feat_extract.5/", "/Convs.1/"]

all_nodes = [n.name for n in model.graph.node]
excluded_nodes = [
    n for n in all_nodes if any(n.startswith(prefix) for prefix in exclude_prefixes)
]


# === 8️⃣ Esegui quantizzazione ===
print("\n🚀 Inizio quantizzazione statica...")
quantize_static(
    model_input=onnx_model,
    model_output=onnx_model_quantized,
    calibration_data_reader=calibration_data_reader,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    nodes_to_exclude=excluded_nodes,
    #per_channel=True,
    reduce_range=True,
    #extra_options={
    #      "WeightSymmetric": True,
    #      "ActivationSymmetric": False,
    #      "CalibMovingAverage": True,
    # }
    #calibrate_method=CalibrationMethod.Distribution, 
    #p_types_to_exclude=["Gemm"]
)
print(f"✅ Modello quantizzato salvato in: {onnx_model_quantized}")

