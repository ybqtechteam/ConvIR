import os
import torch
import argparse
from torch.backends import cudnn
from models.ConvIR import build_net
from train import _train, _train_CL
from eval import _eval
from subset_eval import _subset_eval
from configurations.loader import YAMLLoader
from clearml import Task, TaskTypes
from enum import Enum

class Device(Enum):
    CPU = torch.device('cpu')
    CUDA = torch.device('cuda')
    AUTO = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def from_args(arg):
        if not hasattr(arg, 'device'):
            return Device.AUTO.value
        
        arg = arg.device.lower()
        
        if arg == 'cpu':
            return Device.CPU.value
        elif arg == 'cuda':
            return Device.CUDA.value
        elif arg == 'auto':
            return Device.AUTO.value
        else:
            raise ValueError("Invalid device argument. Choose from 'cpu', 'cuda', or 'auto'.")

def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net(args.type)
    model.to(args.device)
    # print(model)

    if args.mode == 'train' or args.mode == 'concat_train':
        print('Training mode: ', args.phase)
        task = Task.init(project_name=PROJECT_NAME, task_name=args.model_name, task_type=TaskTypes.training)
        task.connect(args)

        if args.phase == 'easy':
            print('Training on easy phase')
            _train(model, args)
        else:
            print('Curriculum learning')
            _train_CL(model, args)

    elif args.mode == 'test':
        print('Simple Evaluation')
        task = Task.init(project_name=PROJECT_NAME, task_name=args.model_name, task_type=TaskTypes.testing)
        task.connect(args)
        _eval(model, args)
    
    elif args.mode == 'subset_test':
        print('Subset evaluation')
        task = Task.init(project_name=PROJECT_NAME, task_name=args.model_name, task_type=TaskTypes.testing)
        task.connect(args)
        _subset_eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file.')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode.', default=False)
    args = parser.parse_args()

    PROJECT_NAME = "VISTA/ConvIR/Dehazing" if not args.debug else "debug"

    args = YAMLLoader(config_path=args.config)
    args.device = Device.from_args(args)

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    command = 'cp ' + 'models/layers.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'models/ConvIR.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'train.py ' + args.model_save_dir
    os.system(command)
    command = 'cp ' + 'main.py ' + args.model_save_dir
    os.system(command)
    command = 'cp -r ' + 'data ' + args.model_save_dir
    os.system(command)
    print(args.__dict__)
    main(args)
