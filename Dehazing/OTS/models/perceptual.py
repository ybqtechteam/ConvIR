
import torch
import torch.nn.functional as F
from torchvision import models


class VGGLoss(torch.nn.Module):
    
    def __init__(self, vgg_model_path="ckpt/vgg16-397923af.pth", device='cuda'):
        super(VGGLoss, self).__init__()
        self.device = device
        self.model = None
        self._load_ckpt(vgg_model_path)
        self.layers = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }
        self.eval()


    def _features(self, x):
        output = {}
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layers:
                output[self.layers[name]] = x
        return list(output.values())

    def forward(self, pred, gt):
        pred = self._features(pred)
        gt = self._features(gt)
        
        loss = []
        for pred_im_feature, gt_feature in zip(pred, gt):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)
    
    def _load_ckpt(self, path):
        model = models.vgg16(pretrained=False)
        model.load_state_dict(torch.load(path))
        model = model.features[:16]
        model.to(self.device)
        for param in model.parameters():
            param.requires_grad = False
        self.model = model


if __name__ == "__main__":
    model = VGGLoss()
    print(model)
    input1 = torch.randn((2, 3, 480, 640)).cuda()
    input2 = torch.randn((2, 3, 480, 640)).cuda()
    loss = model(input1, input2)
    print(loss)

