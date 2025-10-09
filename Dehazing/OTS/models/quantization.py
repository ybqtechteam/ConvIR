# import onnxruntime as ort
import torch


layer = torch.nn.AdaptiveAvgPool2d((1, 1))
x = torch.randn(1, 3, 60, 80)
y = layer(x)
print(y.shape)

print(tuple(x.shape[2:]))
print("-----")


layer2 = torch.nn.AvgPool2d(kernel_size=(60, 80))
_layer2 = torch.nn.functional.avg_pool2d(x, kernel_size=(60, 80))
y2 = layer2(x)
_y2 = _layer2
print(y2.shape)
print(_y2.shape)

print("-----")

layer3 = torch.nn.AdaptiveAvgPool2d((None, 1))
y3 = layer3(x)
print(y3.shape)

print("-----")

layer4 = torch.nn.AvgPool2d(kernel_size=(1, 80)) # metti la dimensione 256 in corrispondenza di quella che vuoi 1
_layer4 = torch.nn.functional.avg_pool2d(x, kernel_size=(1, 80))
y4 = layer4(x)
_y4 = _layer4
print(y4.shape)
print(_y4.shape)