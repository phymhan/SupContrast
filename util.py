from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import os
import sys
import shutil
from glob import glob
from pathlib import Path
from datetime import datetime
import pdb
st = pdb.set_trace

import random

from torch.utils.tensorboard import SummaryWriter

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class TwoCropTransform1:
    """Create two crops of the same image"""
    def __init__(self, transform1):
        self.transform1 = transform1

    def __call__(self, x):
        return [self.transform1(x)]

class TwoCropTransform2:
    """Create two crops of the same image"""
    def __init__(self, transform1, transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x)]

class TwoCropTransform3:
    """Create two crops of the same image"""
    def __init__(self, transform1, transform2, transform3):
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3

    def __call__(self, x):
        return [self.transform1(x), self.transform2(x), self.transform3(x)]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def exclude_bias_and_norm(p):
    return p.ndim == 1


def set_optimizer(opt, model):
    which_optim = getattr(opt, 'optimizer', 'sgd')
    if which_optim == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
    elif which_optim == 'lars':
        optimizer = LARS(model.parameters(), lr=0, weight_decay=opt.weight_decay,
                    weight_decay_filter=exclude_bias_and_norm,
                    lars_adaptation_filter=exclude_bias_and_norm)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def get_hostname():
    try:
        import socket
        return socket.gethostname()
    except:
        return 'unknown'

def get_name_from_args(args):
    name = getattr(args, 'name', Path(args.log_dir).name if hasattr(args, 'log_dir') else 'unknown')
    return name

def print_args(parser, args, is_dict=False, flush=False):
    # args = deepcopy(args)  # NOTE
    if not is_dict and hasattr(args, 'parser'):
        delattr(args, 'parser')
    name = get_name_from_args(args)
    datetime_now = datetime.now()
    message = f"Name: {name} Time: {datetime_now}\n"
    message += f"{os.getenv('USER')}@{get_hostname()}:\n"
    if os.getenv('CUDA_VISIBLE_DEVICES'):
        message += f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}\n"
    message += '--------------- Arguments ---------------\n'
    args_vars = args if is_dict else vars(args)
    for k, v in sorted(args_vars.items()):
        comment = ''
        default = None if parser is None else parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '------------------ End ------------------'
    if flush:
        print(message)

    # save to the disk
    log_dir = Path(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir / 'src', exist_ok=True)
    file_name = log_dir / 'args.txt'
    with open(file_name, 'a+') as f:
        f.write(message)
        f.write('\n\n')

    # save command to disk
    file_name = log_dir / 'cmd.txt'
    with open(file_name, 'a+') as f:
        f.write(f'Time: {datetime_now}\n')
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            f.write('CUDA_VISIBLE_DEVICES=%s ' % os.getenv('CUDA_VISIBLE_DEVICES'))
        f.write('deepspeed ' if getattr(args, 'deepspeed', False) else 'python3 ')
        f.write(' '.join(sys.argv))
        f.write('\n\n')

    # backup train code
    shutil.copyfile(sys.argv[0], log_dir / 'src' / f'{os.path.basename(sys.argv[0])}.txt')

try:
    import wandb
except ImportError:
    wandb = None

import logging
def log(output, flush=True):
    logging.info(output)
    if flush:
        print(output)

def setup_wandb_run_id(log_dir, resume=False):
    # NOTE: if resume, use the existing wandb run id, otherwise create a new one
    os.makedirs(log_dir, exist_ok=True)
    file_path = Path(log_dir) / 'wandb_run_id.txt'
    if resume:
        assert file_path.exists(), 'wandb_run_id.txt does not exist'
        with open(file_path, 'r') as f:
            run_id = f.readlines()[-1].strip()  # resume from the last run
    else:
        run_id = wandb.util.generate_id()
        with open(file_path, 'a+') as f:
            f.write(run_id + '\n')
    return run_id

def setup_wandb(args, project=None, name=None, save_to_log_dir=False, resume=False):
    if wandb is not None:
        name = name or get_name_from_args(args)
        resume = getattr(args, 'resume', False)
        run_id = setup_wandb_run_id(args.log_dir, resume)
        args.wandb_run_id = run_id
        if save_to_log_dir:
            wandb_log_dir = Path(args.log_dir) / 'wandb'
            os.makedirs(wandb_log_dir, exist_ok=True)
        else:
            wandb_log_dir = None
        run = wandb.init(
            project=project or getattr(args, 'wandb_project', 'unknown'),
            name=name,
            id=run_id,
            config=args,
            resume=True if resume else "allow",
            save_code=True,
            dir=wandb_log_dir,
        )
    else:
        run = SummaryWriter(args.log_dir / 'tb_logs')
    
    return run


class logging_file(object):
    def __init__(self, path, mode='a+', time_stamp=True, **kwargs):
        self.path = path
        self.mode = mode
        if time_stamp:
            # self.path = self.path + '_' + time.strftime('%Y%m%d_%H%M%S')
            # self.write(f'{time.strftime("%Y%m%d_%H%M%S")}\n')
            self.write(f'{datetime.now()}\n')
    
    def write(self, line_to_print):
        with open(self.path, self.mode) as f:
            f.write(line_to_print)

    def flush(self):
        pass

    def close(self):
        pass

    def __del__(self):
        pass

def get_last_checkpoint(ckpt_dir, ckpt_ext='.pt', latest=None):
    assert ckpt_ext.startswith('.')
    if latest is None:
        ckpt_path = sorted(glob(os.path.join(ckpt_dir, '*'+ckpt_ext)), key=os.path.getmtime, reverse=True)[0]
    else:
        if not latest.endswith(ckpt_ext):
            latest += ckpt_ext
        ckpt_path = Path(ckpt_dir) / latest
    return ckpt_path