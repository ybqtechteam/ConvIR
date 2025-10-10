import torch
from torch.ao.quantization import quantize_dynamic
from torchvision import transforms
from PIL import Image
from models.ConvIR import build_net

# ----------------- Configurazioni -----------------
VERSION = "small"        # small / base / large
PTKL_PATH = "/home/giovannidistasio/quantizzazione/dehaze/ConvIR/OTS/Best.pkl"
QUANTIZED_PATH = "convIR_dynamic_quantized.pth"

torch.backends.quantized.engine = "fbgemm"
print("⚙️ Using quantized engine:", torch.backends.quantized.engine)

# ----------------- Caricamento modello -----------------
print("📦 Caricamento modello e pesi...")
model = build_net(version=VERSION)
model.eval()

state_dict = torch.load(PTKL_PATH, map_location="cpu", weights_only=True)
if "model" in state_dict:
    state_dict = state_dict["model"]

model.load_state_dict(state_dict, strict=False)
print("✅ Pesi caricati (strict=False)")

# ----------------- Quantizzazione dinamica -----------------
print("⚙️ Avvio quantizzazione dinamica...")

# Applica quantizzazione solo a layer compatibili
# (Conv2d non è supportato in quantizzazione dinamica!)
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},  # quantizza solo layer Linear
    dtype=torch.qint8    # puoi anche provare torch.float16
)

print("✅ Quantizzazione dinamica completata.")

# ----------------- Salvataggio modello -----------------
torch.save(quantized_model.state_dict(), QUANTIZED_PATH)
print(f"💾 Modello quantizzato dinamicamente salvato in '{QUANTIZED_PATH}'")

# ----------------- Test inferenza -----------------
print("🧪 Test inferenza con dummy input...")
dummy_input = torch.randn(1, 3, 480, 640)
with torch.no_grad():
    output = quantized_model(dummy_input)

if isinstance(output, (list, tuple)):
    print("✅ Inferenza completata. Output shapes:", [o.shape for o in output])
else:
    print("✅ Inferenza completata. Output shape:", output.shape)
