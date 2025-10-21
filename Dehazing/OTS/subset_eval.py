import os
import torch
from torchvision.transforms import functional as F
from utils import Adder, SubsetAnalyzer
from data.concat_data_load import test_subset_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time
from pytorch_msssim import ssim
import torch.nn.functional as f
from clearml import Task

def _subset_eval(model, args):
    task = Task.current_task()
    device = args.device
    state_dict = torch.load(args.test_model, weights_only=True, map_location=device)
    model.load_state_dict(state_dict['model'])
    dataloader = test_subset_dataloader(args.test_subset_dir, args.subset_ratio, batch_size=1, num_workers=0)

    analyzer = SubsetAnalyzer(["psnr", "ssim", 'time'], folder_path=args.result_dir)

    model.eval()
    factor = 32

    for n in range(args.number_subsets):
        print(f"----- EVAL RUN {n+1}/{args.number_subsets} -----")
        
        with torch.no_grad():
            psnr_adder = Adder()
            ssim_adder = Adder()
            time_adder = Adder()

            for iter_idx, data in enumerate(dataloader):
                input_img, label_img, name = data
                input_img = input_img.to(device)

                h, w = input_img.shape[2], input_img.shape[3]
                H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
                padh = H-h if h%factor!=0 else 0
                padw = W-w if w%factor!=0 else 0
                input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

                tm = time.time()

                pred = model(input_img)[2]
                pred = pred[:,:,:h,:w]

                elapsed = time.time() - tm
                time_adder(elapsed)

                pred_clip = torch.clamp(pred, 0, 1)

                pred_numpy = pred_clip.squeeze(0).cpu().numpy()
                label_numpy = label_img.squeeze(0).cpu().numpy()

                label_img = (label_img).to(device)
                psnr_val = 10 * torch.log10(1 / f.mse_loss(pred_clip, label_img))
                down_ratio = max(1, round(min(H, W) / 256))	
                ssim_val = ssim(f.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))), 
                                f.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))), 
                                data_range=1, size_average=False)	
                print('%d iter PSNR_dehazing: %.2f ssim: %f' % (iter_idx + 1, psnr_val, ssim_val))
                ssim_adder(ssim_val)

                if args.save_image:
                    folder_path = os.path.join(args.result_dir, f'subset_{n+1}')
                    os.makedirs(folder_path, exist_ok=True)
                    save_name = os.path.join(folder_path, name[0])
                    pred_clip += 0.5 / 255
                    pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                    pred.save(save_name)

                psnr_mimo = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
                psnr_adder(psnr_val)

                print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr_mimo, elapsed))

            print('==========================================================')
            print('The average PSNR is %.2f dB' % (psnr_adder.average()))
            print('The average SSIM is %.4f dB' % (ssim_adder.average()))
            print("Average time: %f" % time_adder.average())

            analyzer.update(psnr_adder.average().item(), ssim_adder.average().item(), time_adder.average())

            task.get_logger().report_scalar("TEST", "PSNR", value=psnr_adder.average(), iteration=n)
            task.get_logger().report_scalar("TEST", "SSIM", value=ssim_adder.average(), iteration=n)
            task.get_logger().report_scalar("TEST", "TIME", value=time_adder.average(), iteration=n)

    df = analyzer.show_csv(save=True)
    analyzer.boxplot(save=True)
    task.get_logger().report_table("Test Subset Results", "Metrics", iteration=0, table_plot=df)
