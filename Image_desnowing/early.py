from colorama import Fore

class EarlyStopping:
    
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
