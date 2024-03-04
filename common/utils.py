import os
#import wandb
import torch
import pprint
import random
import argparse
import numpy as np
from termcolor import colored
import torch.nn as nn
import torch.nn.functional as F

def total_loss(alpha1, alpha2, alpha3, distillation_scr_loss, distillation_cca_loss, cca_loss, scr_loss):

    weighted_distillation = alpha1 * (distillation_scr_loss + distillation_cca_loss)
    weighted_cca_loss = alpha2 * cca_loss
    weighted_scr_loss = alpha3 * scr_loss
    total_loss = weighted_distillation + weighted_cca_loss + weighted_scr_loss
    return total_loss

def compute_prototypes(support_data, support_labels):
    """
    计算支持集数据的原型表示
    参数:
        - support_data (Tensor): 支持集样本的数据，形状为 (num_support, num_channels, height, width)
        - support_labels (Tensor): 支持集样本的标签，形状为 (num_support,)
    返回:
        - prototypes (Tensor): 支持集数据的原型表示，形状为 (num_classes, num_channels, height, width)
    """
    num_classes = torch.unique(support_labels).size(0)
    num_channels, height, width = support_data.size(1), support_data.size(2), support_data.size(3)
    prototypes = torch.zeros(num_classes, num_channels, height, width)
    
    for class_idx in range(num_classes):
        class_samples = support_data[support_labels == class_idx]
        prototypes[class_idx] = torch.mean(class_samples, dim=0)
    
    return prototypes

class DistillKL(nn.Module):
    """KL divergence for distillation"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss

def setup_run(arg_mode='train'):
    args = parse_args(arg_mode=arg_mode)
    pprint(vars(args))

    torch.set_printoptions(linewidth=100)
    args.num_gpu = set_gpu(args)
    args.device_ids = None if args.gpu == '-1' else list(range(args.num_gpu))
    args.save_path = os.path.join(f'checkpoints/{args.dataset}/{args.shot}shot-{args.way}way/-conv', args.extra_dir)
    ensure_path(args.save_path)

    if not args.no_wandb:
        wandb.init(project="my-wandb-project",
                   config=args,
                   save_code=True,
                   name=args.extra_dir)

    if args.dataset == 'miniimagenet':
        args.num_class = 64
    elif args.dataset == 'cub':
        args.num_class = 100
    elif args.dataset == 'tieredimagenet':
        args.num_class = 351
    elif args.dataset == 'cars':
        args.num_class = 130
    elif args.dataset == 'dogs':
        args.num_class = 60
    elif args.dataset == 'flowers':
        args.num_class = 51

    return args

def set_gpu(args):
    if args.gpu == '-1':
        gpu_list = [int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_list = [int(x) for x in args.gpu.split(',')]
        print('use gpu:', gpu_list)
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)

def compute_accuracy(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).type(torch.float).mean().item() * 100.

_utils_pp = pprint.PrettyPrinter()

def pprint(x):
    _utils_pp.pprint(x)

def load_model(model, dir):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(dir)['params']

    if pretrained_dict.keys() == model_dict.keys():  # load from a parallel meta-trained model and all keys match
        print('all state_dict keys match, loading model from :', dir)
        model.load_state_dict(pretrained_dict)
    else:
        print('loading model from :', dir)
        if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
            if 'module' in list(pretrained_dict.keys())[0]:
                pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
            else:
                pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        else:
            pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}  # load from a pretrained model
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
        model.load_state_dict(model_dict)

    return model

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def detect_grad_nan(model):
    for param in model.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print("NaN gradient detected in model.")
            return True
    return False

def by(s):
    '''
    :param s: str
    :type s: str
    :return: bold face yellow str
    :rtype: str
    '''
    bold = '\033[1m' + f'{s:.3f}' + '\033[0m'
    yellow = colored(bold, 'yellow')
    return yellow

def parse_args(arg_mode):
    parser = argparse.ArgumentParser(description='Relational Embedding for Few-Shot Classification (ICCV 2021) cca+mixup')

    ''' about dataset '''
    parser.add_argument('-dataset', type=str, default='cub',
                        choices=['miniimagenet', 'cub', 'tieredimagenet', 'cars', 'dogs', 'flowers'])
    parser.add_argument('-data_dir', type=str, default='/data/datasets_share', help='dir of datasets')

    ''' about training specs '''
    parser.add_argument('-batch', type=int, default=32, help='auxiliary batch size')
    parser.add_argument('-temperature', type=float, default=0.2, metavar='tau', help='temperature for metric-based loss')
    parser.add_argument('-lamb', type=float, default=1.5, metavar='lambda', help='loss balancing term')

    ''' about training schedules '''
    parser.add_argument('-max_epoch', type=int, default=80, help='max epoch to run')
    parser.add_argument('-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('-alpha', type=float, default=1.0, help='alpha in mixup')
    parser.add_argument('-gamma', type=float, default=0.05, help='learning rate decay factor')
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70], help='milestones for MultiStepLR')
    parser.add_argument('-save_all', action='store_true', default=False, help='save models on each epoch')

    ''' about few-shot episodes '''
    parser.add_argument('-way', type=int, default=5, metavar='N', help='number of few-shot classes')
    parser.add_argument('-shot', type=int, default=1, metavar='K', help='number of shots')
    parser.add_argument('-query', type=int, default=15, help='number of query image per class')
    parser.add_argument('-val_episode', type=int, default=200, help='number of validation episode')
    parser.add_argument('-test_episode', type=int, default=2000, help='number of testing episodes after training')

    ''' about SCR '''
    parser.add_argument('-self_method', type=str, default='scr')

    ''' about CCA '''
    parser.add_argument('-temperature_attn', type=float, default=1.0, metavar='gamma', help='temperature for softmax in computing cross-attention')

    ''' about env '''
    parser.add_argument('-gpu', default='0', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')
    parser.add_argument('-extra_dir', type=str, default='test', help='extra dir name added to checkpoint dir')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-no_wandb', action='store_true', help='not plotting learning curve on wandb',
				default=True)
                        #default=arg_mode == 'test')  # train: enable logging / test: disable logging
    args = parser.parse_args()
    return args
