import torch
from torchvision.transforms import functional as F
from data import valid_dataloader, valid_dataloader_CL
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as f
from data.concat_data_load import _valid_concat_dataloader
from pytorch_msssim import ssim


def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'train':
        dataloader = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    elif args.mode == 'concat_train':
        dataloader = _valid_concat_dataloader(args.data_concat_train_dir, batch_size=1, num_workers=0)
    
    model.eval()
    psnr_adder = Adder()
    ssim_adder = Adder()

    with torch.no_grad():
        print('Start Evaluation')
        factor = 32
        for _, data in enumerate(dataloader):
            input_img, label_img, _ = data
            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            pred = model(input_img)[2]
            pred = pred[:,:,:h,:w]

            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)

            psnr_adder(psnr)

            label_img = label_img.cuda()
            down_ratio = max(1, round(min(H, W) / 256))	
            ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))), 
                            f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False)	
            
            ssim_adder(ssim_val)

    print('\n')
    model.train()
    
    #* return average PSNR and SSIM
    # return psnr_adder.average(), ssim_adder.average()
    return psnr_adder.average()



























def _valid_CL(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ots = valid_dataloader_CL(args.data_curriculum_learning_dir, batch_size=1, num_workers=0, phase=args.phase)
    model.eval()
    psnr_adder = Adder()

    with torch.no_grad():
        print('Start Evaluation')
        factor = 32
        for _, data in enumerate(ots):
            input_img, label_img, _ = data
            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            pred = model(input_img)[2]
            pred = pred[:,:,:h,:w]

            pred_clip = torch.clamp(pred, 0, 1)
            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)

            psnr_adder(psnr)

    print('\n')
    model.train()
    return psnr_adder.average()
