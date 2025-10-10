import torch
from torch.ao.quantization import prepare, convert, get_default_qconfig

torch.backends.quantized.engine = 'fbgemm'
print("engines:", torch.backends.quantized.supported_engines)
print("using:", torch.backends.quantized.engine)

# 1️⃣ Creo un Conv2d float
conv = torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1).cpu()

# 2️⃣ Assegno un qconfig
conv.qconfig = get_default_qconfig('fbgemm')

# 3️⃣ Preparo e converto
prepare(conv, inplace=True)

# 4️⃣ Calibrazione simulata
x = torch.randn(1, 3, 16, 16)
with torch.no_grad():
    conv(x)

# 5️⃣ Conversione finale
qconv = convert(conv, inplace=False)

# 6️⃣ Test inferenza
with torch.no_grad():
    out = qconv(x)
print("✅ quantized conv test ran OK. output shape:", out.shape)
