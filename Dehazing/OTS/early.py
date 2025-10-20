from colorama import Fore

class EarlyStoppingSingleMetric:
    
    def __init__(self, patience=15, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_psnr = None
        self.early_stop = False

    def __call__(self, val_psnr):
        if self.best_psnr is None:
            self.best_psnr = val_psnr

        elif val_psnr > self.best_psnr + self.min_delta:
            # Miglioramento significativo
            self.best_psnr = val_psnr
            self.counter = 0
            print(Fore.YELLOW + "EarlyStopping reset" + Fore.RESET)
            
        else:
            # Non migliora abbastanza
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class EarlyStoppingDoubleMetric:
    
    def __init__(self, patience=15, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.psnr_counter = 0
        self.ssim_counter = 0
        self.best_psnr = None
        self.best_ssim = None
        self.psnr_early_stop = False
        self.ssim_early_stop = False

    def forward_psnr(self, val_psnr):
        if self.best_psnr is None:
            self.best_psnr = val_psnr

        elif val_psnr >= self.best_psnr + self.min_delta:
            self.best_psnr = val_psnr
            self.psnr_counter = 0
            print(Fore.YELLOW + "EarlyStopping PSNR reset" + Fore.RESET)
            
        else:
            self.psnr_counter += 1
            if self.psnr_counter >= self.patience:
                self.psnr_early_stop = True
                print(Fore.RED + "EarlyStopping PSNR triggered" + Fore.RESET) if self.psnr_counter == self.patience else None
    
    def forward_ssim(self, val_ssim):
        if self.best_ssim is None:
            self.best_ssim = val_ssim

        elif val_ssim >= self.best_ssim + self.min_delta:
            self.best_ssim = val_ssim
            self.ssim_counter = 0
            print(Fore.YELLOW + "EarlyStopping SSIM reset" + Fore.RESET)
            
        else:
            self.ssim_counter += 1
            if self.ssim_counter >= self.patience:
                self.ssim_early_stop = True
                print(Fore.RED + "EarlyStopping SSIM triggered" + Fore.RESET) if self.ssim_counter == self.patience else None
    
    def is_stopped(self):
        return self.psnr_early_stop and self.ssim_early_stop
    


if __name__ == "__main__": 
    es = EarlyStoppingDoubleMetric(patience=3, min_delta=0.1)

    fake_psnr = [25, 25.05, 25.3, 26, 25.13, 25.14, 25.15]
    fake_ssim = [0.80, 0.9, 0.83, 0.84, 0.85, 0.86, 0.86]

    for epoch, (p, s) in enumerate(zip(fake_psnr, fake_ssim), 1):
        print(f"\nEpoch {epoch}")
        es(p)
        es.forward_ssim(s)
        if es.is_stopped():
            print(Fore.RED + f"Training stopped at epoch {epoch}" + Fore.RESET)
            break
