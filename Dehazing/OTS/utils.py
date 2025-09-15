import time, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
    return lr


def summary(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")






class SubsetAnalyzer:
    
    def __init__(self, columns: List = None, folder_path: str = None):
        self.folder_path = Path(folder_path).joinpath('plots') if folder_path else None
        self.folder_path.mkdir(parents=True, exist_ok=True) if folder_path else None
        self.df = pd.DataFrame(columns=columns) if columns else None

    
    def update(self, *args):
        self.df.loc[len(self.df)] = [*args]
    

    def show_csv(self, precision: int = 2, save: bool = False):
        df = self.df.copy()

        mean = []
        std = []

        for metric in df.columns:
            mean.append(df[metric].mean())
            std.append(df[metric].std())
        
        df.loc[len(df)] = mean
        df.loc[len(df)] = std
        df = df.round(precision)
        df.index = pd.Index([f'Subset: {i+1}' for i in range(len(df) - 2)] + ["Mean", "Std"])

        print(f"\n{df}")

        if save:
            df.to_csv(self.folder_path / "subset_analysis.csv", index=True)

        return df

    
    def boxplot(self, file: str = None, save: bool = False):
        if file is not None:
            df = pd.read_csv(file)
            df = df.drop(df.columns[0], axis=1)
        else:
            df = self.df.copy()
        
        df = df.drop(df.columns[-1], axis=1) if 'time' in df.columns else df

        for metric in df.columns.to_list():
            df_to_plot = df[[metric]]
            pd.plotting.boxplot(df_to_plot, column=df_to_plot.columns.to_list(), grid=True)
            
            if save:
                plt.savefig(self.folder_path / f"boxplot_{metric}.png")
            
            plt.show()
            plt.clf()
        
        plt.close('all')
    

    @staticmethod
    def boxplot_compare_per_metric(dataframes_path: List[str], save: bool = False):
        metrics = pd.read_csv(dataframes_path[0]).columns.to_list()
        metrics.remove('time') if 'time' in metrics else None
        metrics.pop(0)
        
        for m in metrics:
            dfs = [pd.read_csv(f, usecols=[m]) for f in dataframes_path]
            df = pd.concat(dfs, axis=1)
            df = df.iloc[:-2]
            df.columns = [f"Model {i+1}" for i in range(len(dfs))]

            pd.plotting.boxplot(df, column=df.columns.to_list(), grid=True)
            plt.title(f'Boxplot Comparison for {m}')
            if save:
                plt.savefig(f"boxplot_comparison_{m}.png")
            plt.show()
            plt.clf()
        
        plt.close('all')









if __name__ == "__main__":
    # from models.ConvIR import build_net

    # model = build_net('large')
    # summary(model)

    # analyzer = SubsetAnalyzer(folder_path='.', columns=["psnr", "ssim", "time"])
    # analyzer.update(25, 1, 5)
    # analyzer.update(30, 0.5, 10)
    # analyzer.update(35, 0, 15)
    # analyzer.show_csv(save=True)
    # analyzer.boxplot(save=True)

    path = ['RESULTS/EXP1/baseline/ConvIR/test/plots/subset_analysis.csv', 
            'RESULTS/EXP1/synthetic_dataset/ConvIR/test/plots/subset_analysis.csv']
    
    SubsetAnalyzer.boxplot_compare_per_metric(path, save=True)