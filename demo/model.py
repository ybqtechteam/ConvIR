import torch, cv2, numpy as np
import PIL.Image as Image
import torch.nn as nn
from Image_deraining.models.ConvIR import build_net as build_derain_net
from Dehazing.OTS.models.ConvIR import build_net as build_dehaze_net
from Image_desnowing.models.ConvIR import build_net as build_desnow_net
from typing import List
from numpy import ndarray
from .data import PairResize, PairToTensor
import torch.nn.functional as f
from torchvision.transforms import functional as F
import enum



class MODEL_TYPE(enum.Enum):
    DERAIN = "Image_deraining/RESULTS/small_version_LHP_RR_LW/ConvIR/train/Best.pkl"
    DEHAZE = 'Dehazing/OTS/RESULTS/EXP1/synthetic_dataset/ConvIR/OTS/Best.pkl'
    DESNOW = "Image_desnowing/RESULTS/small_version_RV_RS85_RS10K_resume/ConvIR/train/Best.pkl"
    DEHAZE_BASE = "Dehazing/OTS/synthetic_dataset_base/ConvIR/OTS/Best.pkl"



class GeneralConvIR():

    def __init__(self, model_path: str):
        self.forward = None
        self.factor = 32
        self.h, self.w = None, None
        self.model = None
        self.H, self.W = None, None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load(model_path)
        self.t1 = PairResize((640, 480))
        self.t2 = PairToTensor()
        

    def _preprocessing(self, img: ndarray) -> torch.Tensor:
        img = Image.fromarray(img)
        self.W, self.H = img.size
        x = self.t1(img, None)
        x: torch.Tensor = self.t2(x, None)
        x = x.unsqueeze(0).to(self.device)
        self.h, self.w = x.shape[2], x.shape[3]
        H, W = ((self.h+self.factor)//self.factor)*self.factor, ((self.w+self.factor)//self.factor*self.factor)
        padh = H-self.h if self.h%self.factor !=0 else 0
        padw = W-self.w if self.w%self.factor !=0 else 0
        x = f.pad(x, (0, padw, 0, padh), 'reflect')
        return x 
    
    def _postprocessing(self, x: List[torch.Tensor]) -> Image:
        pred = x[2]
        pred = pred[:,:,:self.h,:self.w]
        pred_clip = torch.clamp(pred, 0, 1)
        pred_clip += 0.5 / 255
        pred: Image = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
        # pred = pred.resize((self.W, self.H), Image.BICUBIC)
        return pred
    
    # TODO: implement onnx loading
    def _load(self, model_path: str) -> nn.Module:
        if model_path.endswith('.onnx'):
            self.forward = self._forward_onnx
            pass
        else:
            model = self._build_net()
            model.to(self.device)
            state_dict = torch.load(model_path, weights_only=True, map_location=self.device)
            model.load_state_dict(state_dict['model'])
            model.eval()
            self.model = model
            self.forward = self._forward_torch
    
    def _forward_torch(self, x: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            out = self.model(x)
        return out
    
    # TODO: implement onnx forward
    def _forward_onnx(self, x: torch.Tensor) -> List[torch.Tensor]:
        pass

    def _build_net(self):
        raise NotImplementedError
    
    def __call__(self, x):
        x = self._preprocessing(x)
        x = self.forward(x)
        x = self._postprocessing(x)
        return x

class DerainConvIR(GeneralConvIR):

    def _build_net(self):
        return build_derain_net(num_res=4)


class DehazeConvIR(GeneralConvIR):

    def __init__(self, model_path: str, version: str = 'small'):
        self.version = version
        super().__init__(model_path)

    def _build_net(self):
        return build_dehaze_net(self.version)


class DesnowConvIR(GeneralConvIR):

    def _build_net(self):
        return build_desnow_net('small')




