import logging
import os
import pickle
import time
import math
import random
import json

import torch
from torch import nn, optim
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np


from dataset import IdxDataset, ClassDataset, ConcatDataset
from dist_utils import gather_from_all, init_distributed_mode, get_rank, is_main_process, get_world_size
from util import AverageMeter

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
_logger = logging.getLogger('train')

def load_top5(args):
    if os.path.isfile(args.top5_path):
        print('Loading top5 dict')
        with open(args.top5_path, 'rb') as f:
            return pickle.load(f)

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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main_worker(args):
    init_distributed_mode(args)

    device = torch.device('cuda')

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = True

    if is_main_process():
        try:
            tb_logger = SummaryWriter(args.log_dir + args.name)
        except:
            tb_logger = SummaryWriter(args.log_dir + args.name + '1')

    num_tasks = get_world_size()
    global_rank = get_rank()

    _logger.info('Creating dataset')
    dataset = torchvision.datasets.ImageFolder(args.data, Transform(args))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=False)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    sampler.set_epoch(0)

    test_dataset = torchvision.datasets.ImageFolder(args.test_data, Transform(args))
    test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=False)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    sampler.set_epoch(0)

    _logger.info('Creating model')
    model = SimCLR(args).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    lin_clf = nn.Linear(2048, 1000)
    lin_clf = torch.nn.parallel.DistributedDataParallel(lin_clf, device_ids=[args.gpu])
    lin_clf_without_ddp = lin_clf.module

    if (args.checkpoint_path).is_file():
        _logger.info('Loading encoder weights from checkpoint')
        ckpt = torch.load(args.checkpoint_path, map_location='cpu')
        model_without_ddp.load_state_dict(ckpt['model'])

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        _logger.info('Resuming from checkpoint')
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        args.start_epoch = ckpt['epoch']
        lin_clf_without_ddp.load_state_dict(ckpt['model'])
    else:
        args.start_epoch = 0

    optimizer = optim.SGD(lin_clf.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    _logger.info('Starting linear evaluation training')
    train(args, model, lin_clf, optimizer, loader, sampler, device)
    _logger.info('Starting linear evaluation validation')
    validate(args, model, lin_clf, test_loader, test_sampler, device)


def train(args, model, lin_clf, optimizer, loader, sampler, device):
    lin_clf.train()
    model.eval()

    lin_clf_without_ddp = lin_clf.module
    loss_fn = torch.nn.CrossEntropyLoss()

    avg_itr = AverageMeter()
    avg_loss = AverageMeter()
    avg_top1 = AverageMeter()
    avg_top5 = AverageMeter()
    
    scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()
    itr = args.start_epoch * len(loader)

    for epoch in range(args.start_epoch, args.epochs):
        _logger.info(f'Starting training epoch {epoch}')
        sampler.set_epoch(epoch)
        
        for step, (images, labels) in enumerate(loader, start=epoch * len(loader)):
            itr_start = time.time()
            itr += 1
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # lr = adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()

            with torch.no_grad():
                features = model.forward_backbone(images)

            outputs = lin_clf(features.detach())

            loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            outputs = gather_from_all(outputs)
            labels = gather_from_all(labels)

            # update metrics
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

            itr_end = time.time()
            itr_time = itr_end - itr_start
            
            if is_main_process():
                bs = labels.shape[0]
                avg_loss.update(loss.item(), bs)
                avg_itr.update(itr_time)
                avg_top1.update(acc1[0], bs)
                avg_top5.update(acc5[0], bs)

            if step % args.print_freq == 0:
                torch.distributed.reduce(acc1[0].div_(args.world_size), 0)
                torch.distributed.reduce(acc5[0].div_(args.world_size), 0)
                if is_main_process():
                    _logger.info(f'epoch={epoch}, step={step}, loss={loss.item()}, acc1={acc1[0].item()}, acc5={acc5[0].item()}, itr time={itr_time}')
                    stats = dict(epoch=epoch, step=step, #learning_rate=lr,
                                 loss=loss.item(), acc1=acc1.item(), acc5=acc5.item(),
                                 time=int(time.time() - start_time))
                    with open(args.checkpoint_dir / 'stats.txt', 'a') as stats_file:
                        stats_file.write(json.dumps(stats) + "\n")

        if is_main_process():
            # save checkpoint
            _logger.info(f'Saved checkpoint {epoch}')
            state = dict(epoch=epoch + 1, model=lin_clf_without_ddp.state_dict(),
                         optimizer=optimizer.state_dict())
            if (args.checkpoint_dir / 'checkpoint.pth').is_file():
                os.rename(args.checkpoint_dir / 'checkpoint.pth', args.checkpoint_dir / f'checkpoint_{epoch}')
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')

    if is_main_process():
        # save final model
        _logger.info(f'Saved final checkpoint')
        torch.save(dict(classifier=lin_clf_without_ddp.state_dict()),
                    args.checkpoint_dir + args.name + '-resnet50.pth')
    
def validate(args, model, lin_clf, loader, sampler, device):
    lin_clf.eval()
    model.eval()

    lin_clf_without_ddp = lin_clf.module
    loss_fn = torch.nn.CrossEntropyLoss()

    avg_itr = AverageMeter()
    avg_loss = AverageMeter()
    avg_top1 = AverageMeter()
    avg_top5 = AverageMeter()

    start_time = time.time()

    _logger.info(f'Starting validation')
    sampler.set_epoch(args.epoch+1)
    
    for step, (images, labels) in enumerate(loader):
        itr_start = time.time()
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            features = model.forward_backbone(images)
            outputs = lin_clf(features.detach())

        outputs = gather_from_all(outputs)
        labels = gather_from_all(labels)

        loss = loss_fn(outputs, labels)

        # update metrics
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        itr_end = time.time()
        itr_time = itr_end - itr_start
        
        if is_main_process():
            bs = labels.shape[0]
            avg_loss.update(loss.item(), bs)
            avg_itr.update(itr_time)
            avg_top1.update(acc1[0], bs)
            avg_top5.update(acc5[0], bs)

        if step % args.print_freq == 0:
            torch.distributed.reduce(acc1[0].div_(args.world_size), 0)
            torch.distributed.reduce(acc5[0].div_(args.world_size), 0)
            if is_main_process():
                _logger.info(f'Test: step={step}, loss={loss.item()}, acc1={acc1[0].item()}, acc5={acc5[0].item()}, itr time={itr_time}')
                stats = dict(step=step, #learning_rate=lr,
                                loss=loss.item(), acc1=acc1.item(), acc5=acc5.item(),
                                time=int(time.time() - start_time))
                with open(args.checkpoint_dir / 'test_stats.txt', 'a') as stats_file:
                    stats_file.write(json.dumps(stats) + "\n")



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

    def forward(self, y1, y2, neg_images=None, labels=None, neg_labels=None, top5_labels=None):
        r1 = self.backbone(y1)
        r2 = self.backbone(y2)
        if neg_images is not None:
            r3 = self.backbone(neg_images)

        # projection
        z1 = self.projector(r1)
        z2 = self.projector(r2)
        
        if neg_images is not None:
            z3 = self.projector(r3)

        if neg_images is not None:
            loss = self.loss_fn(z1, z2, labels, neg_features=z3, neg_labels=neg_labels)
        elif top5_labels is not None:
            loss = self.loss_fn(z1, z2, labels, top5_labels=top5_labels)
        else:
            loss = self.loss_fn(z1, z2)

        logits = self.onne_head(r1.detach())
        cls_loss = torch.nn.functional.cross_entropy(logits, labels)
        acc = torch.sum(torch.eq(torch.argmax(logits, dim=1), labels)) / logits.size(0)

        loss = loss + cls_loss

        return loss, acc
    
    def forward_backbone(self, y1):
        return self.backbone(y1)
    
    def reset_linear_head(self):
        self.onne_head = nn.Linear(2048, 1000)


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

def supcon_loss(z1, z2, labels=None, neg_features=None, neg_labels=None, top5_labels=None, mask=None, temperature=0.1, base_temperature=0.07, contrast_mode='all'):
    features1 = torch.nn.functional.normalize(z1, dim=1)
    features2 = torch.nn.functional.normalize(z2, dim=1)

    features1 = gather_from_all(features1)
    features2 = gather_from_all(features2)
    labels = gather_from_all(labels)
    if top5_labels is not None:
        top5_labels = gather_from_all(top5_labels)

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

        if top5_labels is not None:
            top5_mask = torch.eq(labels, top5_labels.unsqueeze(-1).permute(1,2,0))
            top5_mask = torch.any(top5_mask, dim=0)

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
    # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast #- logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)

    if top5_labels is not None:
        top5_mask = top5_mask.repeat(anchor_count, contrast_count)

    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    top5_mask = top5_mask * logits_mask

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

        self.transform_lin_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])


    def __call__(self, x):
        y1 = self.transform_lin_eval(x)
        return y1