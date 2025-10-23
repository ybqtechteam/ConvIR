import os
import torch
import argparse
from torch.backends import cudnn
from models.ConvIR import build_net
from train import _train
from eval import _eval
from subset_eval import _subset_eval, _subset_eval_onnx
from configurations.loader import YAMLLoader
from clearml import Task, TaskTypes
from models.quantization import ONNXModel

def main(args):
    cudnn.benchmark = True

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    if args.mode == 'subset_test_onnx':
        task = Task.init(project_name=f"{PROJECT_NAME}/Onnx", task_name=args.model_name, task_type=TaskTypes.testing)
        task.connect(args)
        model = ONNXModel(args.test_model)
        _subset_eval_onnx(model, args)
        return

    model = build_net(args.version)
    
    if torch.cuda.is_available():
        model.cuda()
    
    if args.mode == 'train' or args.mode == 'concat_train':
        print('Training')
        task = Task.init(project_name=PROJECT_NAME, task_name=args.model_name, task_type=TaskTypes.training)
        task.connect(args)
        _train(model, args)

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

    PROJECT_NAME = "VISTA/ConvIR/Desnowing" if not args.debug else "debug"

    args = YAMLLoader(config_path=args.config)

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
