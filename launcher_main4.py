from pathlib import Path
import logging
import os
import uuid
import subprocess
import math

import submitit
import numpy as np
import argparse

from util import get_last_checkpoint

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
_logger = logging.getLogger('train')

parser = argparse.ArgumentParser(description='RotNet Training')
parser.add_argument('--print_freq', type=int, default=50,
                    help='print frequency')
parser.add_argument('--save_freq', type=int, default=100,
                    help='save frequency')
parser.add_argument('--batch_size', type=int, default=512,
                    help='batch_size')
parser.add_argument('--num_workers', type=int, default=16,
                    help='num of workers to use')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of training epochs')

# optimization
parser.add_argument('--learning_rate', type=float, default=0.05,
                    help='learning rate')
parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                    help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')

# model dataset
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--dataset', type=str, default='cifar100',
                    choices=['cifar10', 'cifar100', 'imagenet', 'path'], help='dataset')
parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

# method
parser.add_argument('--method', type=str, default='simclr',
                    choices=['simclr', 'essl', 'essl+diag', 'simclr+all'], help='choose method')

# temperature
parser.add_argument('--temp', type=float, default=0.07,
                    help='temperature for loss function')

# other setting
parser.add_argument('--cosine', action='store_true',
                    help='using cosine annealing')
parser.add_argument('--syncBN', action='store_true',
                    help='using synchronized batch normalization')
parser.add_argument('--warm', action='store_true',
                    help='warm-up for large batch training')
parser.add_argument('--trial', type=str, default='0',
                    help='id for recording multiple runs')

# ========== add args ==========
parser.add_argument('--log_dir', type=str, default='logs/baseline')
parser.add_argument('--no_wandb', action='store_true')
parser.add_argument('--wandb_project', default='simclr-cifar100', type=str)
parser.add_argument('--cuda', default=None, type=str, help='cuda device ids to use')
parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer')
parser.add_argument('--split', action='store_true')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--no_seed', action='store_true')
parser.add_argument('--append_view', action='store_true')
parser.add_argument('--sample_from_original', action='store_true')
parser.add_argument('--pos_view_paths', type=str, default='')
parser.add_argument('--neg_view_paths', type=str, default='')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='essl scale alpha')
parser.add_argument('--uint8', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--resume_from', type=str, default=None)

parser.add_argument('--add_randomcrop', action='store_true', help='scale (0.9, 1), ratio (0.9, 1.1)')
parser.add_argument('--add_randomcrop2', action='store_true', help='scale (0.7, 1), ratio default')
parser.add_argument('--add_randomcrop3', action='store_true', help='scale (0.5, 1), ratio default')

parser.add_argument('--setting', type=str, default='default',
    choices=['default', 'v1=v2=gan', 'v1=basic,v2=gan', 'v1=expert,v2=gan', 'v1=v2=basic,v3=gan', \
        'v1=v2=gan+basic', 'v1=v2=basic+gan', 'v1=v2=basic', 'v3=basic,v1=v2=gan', 'v1=basic,v2=v3=gan'])

# Slurm setting
parser.add_argument('--ngpus-per-node', default=6, type=int, metavar='N',
                    help='number of gpus per node')
parser.add_argument('--nodes', default=5, type=int, metavar='N',
                    help='number of nodes') 
parser.add_argument("--timeout", default=360, type=int, 
                    help="Duration of the job")
parser.add_argument("--partition", default="el8", type=str, 
                    help="Partition where to submit")

parser.add_argument("--exp", default="SimCLR", type=str, 
                    help="Name of experiment")
  

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main4_aimos
        # self._setup_gpu_args()
        main4_aimos.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        # self.args.dist_url = get_init_file(self.args).as_uri()
        ckpt_path = get_last_checkpoint(
            ckpt_dir=os.path.join(self.args.log_dir, 'weights'),
            ckpt_ext='.pth',
        )
        if os.path.isfile(ckpt_path):
            self.args.resume = True
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)
        
    def _setup_gpu_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def get_init_file(args):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(args.job_dir, exist_ok=True)
    init_file = args.job_dir / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


def main():
    opt = parser.parse_args()

    args = opt

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None  # NOTE: commented out since it's imagenet
    
    # if opt.append_view:
    args.pos_view_paths = [s for s in args.pos_view_paths.split(',') if s != '']
    print(f"pos_view_paths = {args.pos_view_paths}")
    args.neg_view_paths = [s for s in args.neg_view_paths.split(',') if s != '']
    print(f"neg_view_paths = {args.neg_view_paths}")

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # log_dir
    opt.log_dir = Path(opt.log_dir)
    opt.job_dir = opt.log_dir
    opt.use_wandb = not opt.no_wandb
    if opt.cuda is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    
    os.makedirs(opt.log_dir, exist_ok=True)
    os.makedirs(opt.log_dir / 'weights', exist_ok=True)

    # args.checkpoint_dir = args.checkpoint_dir / args.exp
    # args.log_dir = args.log_dir / args.exp
    # args.job_dir = args.checkpoint_dir

    # args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # args.log_dir.mkdir(parents=True, exist_ok=True)

    # get_init_file(args)

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus_per_node
    nodes = args.nodes
    timeout_min = args.timeout
    partition = args.partition

    kwargs = {'slurm_gres': f'gpu:{num_gpus_per_node}',}

    executor.update_parameters(
        mem_gb=0,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=1,  # one task per GPU
        cpus_per_task=24,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 6
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    # executor.update_parameters(name=args.exp)

    # args.dist_url = get_init_file(args).as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    _logger.info(f"Submitted job_id: {job.job_id}")


if __name__ == '__main__':
    main()
