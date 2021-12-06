from pathlib import Path
import argparse
import logging
import os
import uuid

import submitit
import numpy as np
#import wandb

from dist_utils import gather_from_all


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
_logger = logging.getLogger('train')

parser = argparse.ArgumentParser(description='Diverse Hypothesis Contrastive Learning')
parser.add_argument('--data', type=Path, metavar='DIR', default="/anonymous/",
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4096, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=4.8, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='/anonymous/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--log-dir', type=str, default='./logs/')
parser.add_argument('--top5-path', type=str, default='./imagenet_resnet50_top10.pkl')
parser.add_argument('--topk', type=int, default=5, help='K top prediction classes to use for mask creation')
parser.add_argument('--lamb', type=float, default=1.0, help='Lambda for kernel regularization between f and g')
parser.add_argument('--name', type=str, default='test')
parser.add_argument('--seed', type=int, default=21)

# submitit parameters
parser.add_argument('--ngpus', default=6, type=int,
                    help='Number of GPUs per node')
parser.add_argument('--nodes', default=1, type=int,
                    help='Number of nodes to run on')
parser.add_argument("--timeout", default=360, type=int, help="Duration of the job")
parser.add_argument("--partition", default="el8", type=str, help="Partition where to submit")

# variations
parser.add_argument('--dim', default=128, type=int)
parser.add_argument('--layer', default=3, type=int)
parser.add_argument('--temp', default=0.1, type=float)

# colored dataset variations
parser.add_argument('--num_colors', default=10, type=int, help='Number of colors for dataset')
parser.add_argument('--std', default=0.0, type=float, help='Std dev of noise added onto the color')

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import train_helper
        self._setup_gpu_args()
        train_helper.main_worker(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file(self.args).as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = self.args.job_dir
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
    args = parser.parse_args()
    args.checkpoint_dir = args.checkpoint_dir / args.name
    args.job_dir = args.checkpoint_dir

    get_init_file(args)

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout
    partition = args.partition

    kwargs = {'slurm_gres': f'gpu:{num_gpus_per_node}',}

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=16,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 6
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name=args.name)

    args.dist_url = get_init_file(args).as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    _logger.info("Submitted job_id:", job.job_id)


if __name__ == '__main__':
    main()
