import torch
from models.ConvIR import ConvIR  # importa il tuo modello
import pickle
from models.ConvIR import build_net


model = build_net("small")

model.eval()

state = torch.load("/home/giovannidistasio/vista/models/ConvIR/Image_desnowing/small_version_RV_RS85_RS10K_resume/ConvIR/train/Best.pkl", weights_only=True, map_location="cpu")

model.load_state_dict(state['model'])

# 2️⃣ Crea un input di esempio (dummy) con la forma corretta
dummy_input = torch.randn(1, 3, 480, 640)  # batch=1, canali=3, H=480, W=640

# 3️⃣ Esporta in ONNX
torch.onnx.export(
    model,                               # modello PyTorch
    dummy_input,                         # input di esempio
    "convIR_snow.onnx",                       # nome file di output
    input_names=["input"],               # nome simbolico dell’input
    output_names=["output0"],  # nome simbolico dell’output
    opset_version=11,                    
    dynamic_axes={                       # supporta dimensioni dinamiche per batch
        "input": {0: "batch_size"},
        "output0": {0: "batch_size"}
        # "output1": {0: "batch_size"},
        # "output2": {0: "batch_size"}
    },
    #operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
)

print("✅ Modello esportato correttamente in convIR.onnx")
