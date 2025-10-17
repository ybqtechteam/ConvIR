import onnx

model = onnx.load("convIR.onnx")

print("Nodi del decoder:")
for i, node in enumerate(model.graph.node):
    # Se il nome contiene 'decoder'
    if node.name and "" in node.name.lower():
        print(f"{i + 1}: {node.name} ({node.op_type})")
