import os
import torch
import argparse
from torch.backends import cudnn
from models.ConvIR import build_net
from train import _train, _train_CL
from eval import _eval
from subset_eval import _subset_eval
from configurations.loader import YAMLLoader

def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = build_net(args.type)
    # print(model)

    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train' or args.mode == 'concat_train':
        print('Training mode: ', args.phase)
        
        if args.phase == 'easy':
            print('Training on easy phase')
            _train(model, args)
        else:
            print('Curriculum learning')
            _train_CL(model, args)

    elif args.mode == 'test':
        print('Simple Evaluation')
        _eval(model, args)
    
    elif args.mode == 'subset_test':
        print('Subset evaluation')
        _subset_eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
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
    print(args)
    main(args)
