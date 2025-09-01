
def summary(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")




if __name__ == "__main__":
    from models.ConvIR import build_net

    model = build_net('large')
    summary(model)
