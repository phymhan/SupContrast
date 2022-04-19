from pathlib import Path
import argparse
import sys
import random
import time
import json
import math
import logging
import os

from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle
#import wandb

from dist_utils import gather_from_all
from dataset import IdxDataset

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
parser.add_argument('--checkpoint-dir', default='./results/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--log-dir', type=str, default='./logs/')                    
parser.add_argument('--name', type=str, default='test')


# variations
parser.add_argument('--dim', default=128, type=int)
parser.add_argument('--layer', default=3, type=int)
parser.add_argument('--temp', default=0.1, type=float)


def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    args.rank = 0
    args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    #wandb.init(project=args.name)
    args.rank += gpu
            
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    args.checkpoint_dir = args.checkpoint_dir / args.name
    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    
    _logger.info('Creating model')
    model = torchvision.models.resnet50(pretrained=True).cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    model.eval()

    _logger.info('Creating dataset')
    dataset = IdxDataset('imagenet', args.data, Transform(args))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=False)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    sampler.set_epoch(0)

    top5_dict = {}
    logits_dict = {}
    logits_renorm_dict = {}

    file_path = os.path.join(args.checkpoint_dir, 'imagenet_resnet50_top5.pkl')
    logits_file_path = os.path.join(args.checkpoint_dir, 'imagenet_resnet50_logits.pkl')
    logits_renorm_file_path = os.path.join(args.checkpoint_dir, 'imagenet_resnet50_logits_renorm.pkl')

    if os.path.isfile(file_path):
        print('Loading top5 dict')
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    _logger.info('Starting top5 collection')
    # building a dict of index => top 5 predicted labels without the ground truth
    with torch.no_grad():
        for step, (images, labels, idxs) in enumerate(tqdm(loader)):
            itr_start = time.time()
            images = images.cuda(gpu, non_blocking=True)
            labels = labels
            idxs = idxs

            with torch.cuda.amp.autocast():
                outputs = model(images)
            
            outputs = outputs.float().cpu()
            
            outputs = gather_from_all(outputs)
            labels_all = gather_from_all(labels)
            
            idxs_all = gather_from_all(idxs)
            
            preds_top5 = torch.topk(outputs, 5)[1]
            preds_top5_all = gather_from_all(preds_top5)

            for i in range(len(preds_top5_all)):
                top5_nogt = preds_top5_all[i][labels_all[i]!=preds_top5_all[i]]
                top5_dict[int(idxs_all[i])] = top5_nogt
                logits_dict[int(idxs_all[i])] = outputs[i]

                outputs_renorm = outputs[i].clone()
                outputs_renorm = outputs_renorm[torch.arange(outputs_renorm.size(0)) != labels_all[i]].softmax(-1)
                outputs_renorm = torch.cat([outputs_renorm[:labels_all[i]], torch.Tensor([1.0]), outputs_renorm[:labels_all[i]]])

                logits_renorm_dict[idxs_all[i]] = outputs_renorm

        itr_end = time.time()
        itr_time = itr_end - itr_start

        if step % args.print_freq == 0:
            if args.rank == 0:
                _logger.info(f'step={step}, itr time={itr_time}')
    
    _logger.info('Writing top5 to dict')
    # save
    
    if args.rank == 0:
        with open(file_path, 'wb') as f:
            pickle.dump(top5_dict, f, pickle.HIGHEST_PROTOCOL)

        with open(logits_file_path, 'wb') as f:
            pickle.dump(logits_dict, f, pickle.HIGHEST_PROTOCOL)

        with open(logits_renorm_file_path, 'wb') as f:
            pickle.dump(logits_renorm_dict, f, pickle.HIGHEST_PROTOCOL)


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

    def forward(self, y1, y2, labels):
        r1 = self.backbone(y1)
        r2 = self.backbone(y2)

        # projection
        z1 = self.projector(r1)
        z2 = self.projector(r2)

        loss = self.loss_fn(z1, z2, labels)

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

def supcon_loss(z1, z2, labels, mask=None, temperature=0.1, base_temperature=0.07, contrast_mode='all'):
    features1 = gather_from_all(z1)
    features2 = gather_from_all(z2)
    labels = gather_from_all(labels)

    features1 = torch.nn.functional.normalize(features1, dim=1)
    features2 = torch.nn.functional.normalize(features2, dim=1)

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

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature)
    # for numerical stability
    #logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast #- logits_max.detach()

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
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

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

        self.transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])


    def __call__(self, x):
        return self.transform_val(x)


if __name__ == '__main__':
    main()
