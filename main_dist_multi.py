from pathlib import Path
import argparse
import sys
import random
import time
import json
import math
import logging
import os
import uuid


from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import pickle
import submitit
import numpy as np
#import wandb

from dist_utils import gather_from_all, init_distributed_mode, get_rank, is_main_process, get_world_size
from dataset import IdxDataset, ClassDataset

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
_logger = logging.getLogger('train')

parser = argparse.ArgumentParser(description='Supervised Contrastive Learning with Negative Boosting')
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
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='/anonymous/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--log-dir', type=str, default='./logs/')
parser.add_argument('--top5-path', type=str, default='./imagenet_top5.pkl')
parser.add_argument('--name', type=str, default='test')

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

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):

        self._setup_gpu_args()
        main_worker(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file(args).as_uri()
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

def load_top5(args):
    if os.path.isfile(args.top5_path):
        print('Loading top5 dict')
        with open(args.top5_path, 'rb') as f:
            return pickle.load(f)

def main_worker(args):
    init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True

    if is_main_process():
        try:
            tb_logger = SummaryWriter(args.log_dir + args.name)
        except:
            tb_logger = SummaryWriter(args.log_dir + args.name + '1')

    _logger.info('Loading top5 dict')
    top5_dict = load_top5(args)

    num_tasks = get_world_size()
    global_rank = get_rank()

    _logger.info('Creating dataset')
    dataset = IdxDataset('imagenet', args.data, Transform(args))
    neg_dataset = ClassDataset('imagenet', args.data, transform=Transform(args), is_train=True)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, drop_last=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler, drop_last=True)


    _logger.info('Creating model')
    model = SimCLR(args).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    optimizer = LARS(model.parameters(), lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=exclude_bias_and_norm,
                     lars_adaptation_filter=exclude_bias_and_norm)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        _logger.info('Resuming from checkpoint')
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model_without_ddp.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0


    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()

    _logger.info('Starting training')
    for epoch in range(start_epoch, args.epochs):
        _logger.info(f'Starting training epoch {epoch}')
        sampler.set_epoch(epoch)

        for step, ((y1, y2), labels, idxs) in enumerate(loader, start=epoch * len(loader)):
            itr_start = time.time()
            y1 = y1.to(device, non_blocking=True)
            y2 = y2.to(device, non_blocking=True)

            # Get aggregate top 5 samples
            top5 = torch.cat([top5_dict[int(i)] for i in idxs])
            top5 = torch.unique(top5)
            # Sampling (batch_size - 1) number of negative samples
            neg_images = neg_dataset.__getitem__(labels=top5, num_imgs=idxs.shape[0]-1)
            neg_images = neg_images.to(device, non_blocking=True)
            

            lr = adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, acc = model.forward(y1, y2, neg_images, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            itr_end = time.time()
            itr_time = itr_end - itr_start
            
            if is_main_process():
                tb_logger.add_scalar('loss', loss.item())
                tb_logger.add_scalar('acc', acc.item())

            if step % args.print_freq == 0:
                torch.distributed.reduce(acc.div_(args.world_size), 0)
                if is_main_process():
                    _logger.info(f'epoch={epoch}, step={step}, loss={loss.item()}, acc={acc.item()}, itr time={itr_time}')
                    stats = dict(epoch=epoch, step=step, learning_rate=lr,
                                 loss=loss.item(), acc=acc.item(),
                                 time=int(time.time() - start_time))
                    with open(args.checkpoint_dir / 'stats.txt', 'a') as stats_file:
                        stats_file.write(json.dumps(stats) + "\n")

        if is_main_process():
            # save checkpoint
            _logger.info(f'Saved checkpoint {epoch}')
            state = dict(epoch=epoch + 1, model=model_without_ddp.state_dict(),
                         optimizer=optimizer.state_dict())
            if (args.checkpoint_dir / 'checkpoint.pth').is_file():
                os.rename(args.checkpoint_dir / 'checkpoint.pth', args.checkpoint_dir / f'checkpoint_{epoch}')
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')

    if is_main_process():
        # save final model
        _logger.info(f'Saved final checkpoint')
        torch.save(dict(backbone=model_without_ddp.backbone.state_dict(),
                        projector=model_without_ddp.projector.state_dict(),
                        head=model_without_ddp.onne_head.state_dict()),
                    args.checkpoint_dir + args.name + '-resnet50.pth')

        # torch.save(dict(backbone=model.module.backbone.state_dict(),
        #                 projector=model.module.projector.state_dict(),
        #                 head=model.module.onne_head.state_dict()),
        #             args.checkpoint_dir / (str(epoch) + '_checkpoint.pth'))


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.learning_rate  # * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr





class SimCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048] * self.args.layer + [self.args.dim]
        # sizes = [2048, 2048, 2048, self.args.dim]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[-1]))
        self.projector = nn.Sequential(*layers)

        self.onne_head = nn.Linear(2048, 1000)
        self.loss_fn = supcon_loss

    def forward(self, y1, y2, neg_images, labels):
        r1 = self.backbone(y1)
        r2 = self.backbone(y2)
        r3 = self.backbone(neg_images)

        # projection
        z1 = self.projector(r1)
        z2 = self.projector(r2)
        z3 = self.projector(r3)

        loss = self.loss_fn(z1, z2, labels, neg_features=z3)

        logits = self.onne_head(r1.detach())
        cls_loss = torch.nn.functional.cross_entropy(logits, labels)
        acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)

        loss = loss + cls_loss

        return loss, acc


def infoNCE(z1, z2, temperature=0.1):
    z1 = torch.nn.functional.normaze(z1, dim=1)
    z2 = torch.nn.functional.normaze(z2, dim=1)
    z1 = gather_from_all(z1)
    z2 = gather_from_all(z2)
    logits = z1 @ z2.T
    logits /= temperature
    n = z2.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

def supcon_loss(z1, z2, labels, neg_features=None, mask=None, temperature=0.1, base_temperature=0.07, contrast_mode='all'):
    features1 = gather_from_all(z1)
    features2 = gather_from_all(z2)
    neg_features = gather_from_all(neg_features)
    labels = gather_from_all(labels)

    features1 = torch.nn.functional.normalize(features1, dim=1)
    features2 = torch.nn.functional.normalize(features2, dim=1)
    neg_features = torch.nn.functional.normalize(neg_features, dim=1)

    device = (torch.device('cuda')
                if features1.is_cuda
                else torch.device('cpu'))

    batch_size = features1.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = 2
    contrast_feature = torch.cat([features1, features2], dim=0)
    if contrast_mode == 'one':
        anchor_feature = features1
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # If negative features provided separately, add to contrast feature
    if neg_features is not None:
        contrast_feature = torch.cat([contrast_feature, neg_features], dim=0)

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature)
    # for numerical stability
    #logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast #- logits_max.detach()

    if neg_features is not None:
        logits, neg_logits = torch.split(logits, [anchor_feature.shape[0], neg_features.shape[0]], dim=-1)

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    if neg_features is not None:
        # use separate negatives provided
        exp_logits = torch.exp(neg_logits)
        
    else:
        # use negatives within batch
        exp_logits = torch.exp(logits) * logits_mask
    
    # Adding numerator to denominator for normalization
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + torch.exp(logits))
    
    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss


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


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self, args):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.transform_supcon = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])


    def __call__(self, x):
        y1 = self.transform_supcon(x)
        y2 = self.transform_supcon(x)
        return y1, y2


if __name__ == '__main__':
    main()
