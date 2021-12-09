import logging
import os
import pickle
import time
import math
import random
import json
import copy
from itertools import chain

import torch
from torch import nn, optim
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np


from dataset import ColoredDataset
from dist_utils import gather_from_all, init_distributed_mode, get_rank, is_main_process, get_world_size
from util import AverageMeter

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
_logger = logging.getLogger('train')

def main_worker(args):
    init_distributed_mode(args)

    device = torch.device('cuda')

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = True
    tb_logger = None
    if is_main_process():
        try:
            tb_logger = SummaryWriter(args.log_dir + args.name)
        except:
            tb_logger = SummaryWriter(args.log_dir + args.name + '1')

    # num_tasks = get_world_size()
    # global_rank = get_rank()

    _logger.info('Creating dataset')
    og_dataset_train = datasets.MNIST(args.data, train=True, download=True, transform=Transform(args))
    og_dataset_test = datasets.MNIST(args.data, train=False, download=True, transform=Transform(args))

    train_dataset = ColoredDataset(og_dataset_train, classes=args.num_colors, colors=[0, 1], std=args.color_std, color_labels=torch.arange(args.num_colors))
    test_perm = torch.randperm(args.num_colors)
    test_dataset = ColoredDataset(og_dataset_test, classes=args.num_colors, colors=train_dataset.colors[test_perm], std=args.color_std, color_labels=torch.arange(args.num_colors)[test_perm])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=False)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=train_sampler)

    train_sampler.set_epoch(0)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, drop_last=False)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=test_sampler)

    test_sampler.set_epoch(0)

    # Saving colors and random permutation for test dataset
    torch.save({
        'train_colors': train_dataset.colors,
        'test_colors': test_dataset.colors,
        'test_perm': test_perm,
    }, args.checkpoint_dir / 'colors.pt')

    stage = 'stage1'
    _logger.info(f'Creating {stage} model')
    model_stage1 = SimCLR(args, stage=stage).to(device)
    model_stage1 = nn.SyncBatchNorm.convert_sync_batchnorm(model_stage1)
    model_stage1 = torch.nn.parallel.DistributedDataParallel(model_stage1, device_ids=[args.gpu])
    model_stage1_without_ddp = model_stage1.module

    # optimizer = LARS(model.parameters(), lr=0, weight_decay=args.weight_decay,
    #                  weight_decay_filter=exclude_bias_and_norm,
    #                  lars_adaptation_filter=exclude_bias_and_norm)
    optimizer = optim.SGD(model_stage1.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / f'{stage}_checkpoint.pth').is_file():
        _logger.info('Resuming from checkpoint')
        ckpt = torch.load(args.checkpoint_dir / f'{stage}_checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model_stage1_without_ddp.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    # Stage 1 training (f training)
    if args.train_stage1:
        train(args, model_stage1, optimizer, train_loader, train_sampler, start_epoch, stage, device, tb_logger)
        test(args, model_stage1, test_loader, stage, device, tb_logger)

    # Stage 2 training (g training with f frozen)
    stage = 'stage2'
    _logger.info(f'Creating {stage} model')
    model_stage2 = SimCLR(args, stage=stage, stage1_backbone=model_stage1_without_ddp.backbone1, stage1_projector=model_stage1_without_ddp.projector1).to(device)
    model_stage2 = nn.SyncBatchNorm.convert_sync_batchnorm(model_stage2)
    model_stage2 = torch.nn.parallel.DistributedDataParallel(model_stage2, device_ids=[args.gpu])
    model_stage2_without_ddp = model_stage2.module

    optimizer = optim.SGD(model_stage2.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / f'{stage}_checkpoint.pth').is_file():
        _logger.info('Resuming from checkpoint')
        ckpt = torch.load(args.checkpoint_dir / f'{stage}_checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model_stage2_without_ddp.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0
    
    if args.train_stage2:
        train(args, model_stage2, optimizer, train_loader, train_sampler, start_epoch, stage, device, tb_logger)
        test(args, model_stage2, test_loader, stage, device, tb_logger)
    
    # Stage 3 (h training with f, g frozen)
    stage = 'stage3'
    _logger.info(f'Creating {stage} model')
    model_stage3 = SimCLR(args, stage=stage, 
                          stage1_backbone=model_stage1_without_ddp.backbone1, stage1_projector=model_stage1_without_ddp.projector1,
                          stage2_backbone=model_stage2_without_ddp.backbone1, stage2_projector=model_stage2_without_ddp.projector1).to(device)
    model_stage3 = nn.SyncBatchNorm.convert_sync_batchnorm(model_stage3)
    model_stage3 = torch.nn.parallel.DistributedDataParallel(model_stage3, device_ids=[args.gpu])
    model_stage3_without_ddp = model_stage3.module

    optimizer = optim.SGD(model_stage3.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / f'{stage}_checkpoint.pth').is_file():
        _logger.info('Resuming from checkpoint')
        ckpt = torch.load(args.checkpoint_dir / f'{stage}_checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model_stage3_without_ddp.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    # Stage 3 training    
    if args.train_stage3:
        train(args, model_stage3, optimizer, train_loader, train_sampler, start_epoch, stage, device, tb_logger)
        test(args, model_stage3, test_loader, stage, device, tb_logger)


def train(args, model, optimizer, train_loader, train_sampler, start_epoch, stage, device, tb_logger):
    model.train()
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    itr = start_epoch * len(train_loader)
    model_without_ddp = model.module

    _logger.info('Starting training')
    for epoch in range(start_epoch, args.epochs):
        _logger.info(f'Starting {stage} training epoch {epoch}')
        if stage == 'stage1':
            train_sampler.set_epoch(epoch)
        elif stage == 'stage2':
            train_sampler.set_epoch(epoch+args.epochs)
        elif stage == 'stage3':
            train_sampler.set_epoch(epoch+args.epochs*2)
        
        for step, ((y1, y2), digit_labels, color_labels) in enumerate(train_loader, start=epoch * len(train_loader)):
            itr_start = time.time()
            itr += 1
            y1 = y1.to(device, non_blocking=True)
            y2 = y2.to(device, non_blocking=True)

            lr = adjust_learning_rate(args, optimizer, train_loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                infonce_loss, reg_loss, clf_loss, digit_acc1, color_acc1 = model.forward(y1, y2, digit_labels, color_labels, temp=args.temp, lamb=args.lamb)
                loss = infonce_loss + reg_loss + clf_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            itr_end = time.time()
            itr_time = itr_end - itr_start
            
            if is_main_process():
                tb_logger.add_scalar(f'loss/{stage}_total', loss.item(), itr)
                tb_logger.add_scalar(f'loss/{stage}_infonce', infonce_loss.item(), itr)
                tb_logger.add_scalar(f'loss/{stage}_reg', reg_loss.item(), itr)
                tb_logger.add_scalar(f'loss/{stage}_clf', clf_loss.item(), itr)
                tb_logger.add_scalar(f'acc/{stage}_digit_acc', digit_acc1.item(), itr)
                tb_logger.add_scalar(f'acc/{stage}_color_acc', color_acc1.item(), itr)

            if step % args.print_freq == 0:
                torch.distributed.reduce(digit_acc1.div_(args.world_size), 0)
                torch.distributed.reduce(color_acc1.div_(args.world_size), 0)
                if is_main_process():
                    _logger.info(f'stage={stage}, epoch={epoch}, step={step}, loss={loss.item()}, digit acc1={digit_acc1.item()}, color acc1={color_acc1.item()}, itr time={itr_time}')
                    stats = dict(stage=stage, epoch=epoch, step=step, learning_rate=lr,
                                 loss=loss.item(), digit_acc1=digit_acc1.item(), color_acc1=color_acc1.item(),
                                 time=int(time.time() - start_time))
                    with open(args.checkpoint_dir / f'{stage}_stats.txt', 'a') as stats_file:
                        stats_file.write(json.dumps(stats) + "\n")

        if is_main_process():
            # save checkpoint
            _logger.info(f'Saved checkpoint {epoch}')
            state = dict(epoch=epoch + 1, model=model_without_ddp.state_dict(),
                         optimizer=optimizer.state_dict())
            if (args.checkpoint_dir / f'{stage}_checkpoint.pth').is_file():
                os.rename(args.checkpoint_dir / f'{stage}_checkpoint.pth', args.checkpoint_dir / f'{stage}_checkpoint_{epoch}')
            torch.save(state, args.checkpoint_dir / f'{stage}_checkpoint.pth')

def test(args, model, test_loader, stage, device, tb_logger):
    model.eval()

    color_metric = AverageMeter()
    digit_metric = AverageMeter()
    infonce_loss_metric = AverageMeter()
    clf_loss_metric = AverageMeter()
    reg_loss_metric = AverageMeter()
    loss_metric = AverageMeter()

    _logger.info(f'Starting testing {stage}')
    
    with torch.no_grad():
        for step, ((y1, y2), digit_labels, color_labels) in enumerate(test_loader):
            y1 = y1.to(device, non_blocking=True)
            y2 = y2.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                infonce_loss, reg_loss, clf_loss, digit_acc1, color_acc1 = model.forward(y1, y2, digit_labels, color_labels, temp=args.temp, lamb=args.lamb)
                loss = infonce_loss + reg_loss + clf_loss
            
                torch.distributed.reduce(loss.div_(args.world_size), 0)
                torch.distributed.reduce(infonce_loss.div_(args.world_size), 0)
                torch.distributed.reduce(reg_loss.div_(args.world_size), 0)
                torch.distributed.reduce(clf_loss.div_(args.world_size), 0)
                torch.distributed.reduce(digit_acc1.div_(args.world_size), 0)
                torch.distributed.reduce(color_acc1.div_(args.world_size), 0)

                bs = y1.shape[0]
                loss_metric.update(loss.item(), bs)
                infonce_loss_metric.update(infonce_loss.item(), bs)
                reg_loss_metric.update(reg_loss.item(), bs)
                clf_loss_metric.update(clf_loss.item(), bs)
                digit_metric.update(digit_acc1.item(), bs)
                color_metric.update(color_acc1.item(), bs)

        if is_main_process():
            tb_logger.add_scalar(f'loss/test/{stage}_total', loss_metric.avg)
            tb_logger.add_scalar(f'loss/test/{stage}_infonce', infonce_loss_metric.avg)
            tb_logger.add_scalar(f'loss/test/{stage}_reg', reg_loss_metric.avg)
            tb_logger.add_scalar(f'loss/test/{stage}_clf', clf_loss_metric.avg)
            tb_logger.add_scalar(f'acc/test/{stage}_digit_acc', digit_metric.avg)
            tb_logger.add_scalar(f'acc/test/{stage}_color_acc', color_metric.avg)

            _logger.info(f'TEST stage={stage}, loss={loss_metric.avg}, digit acc1={digit_metric.avg}, color acc1={color_metric.avg}')
            stats = dict(stage=stage, loss=loss_metric.avg, digit_acc1=digit_metric.avg, color_acc1=color_metric.avg)
            with open(args.checkpoint_dir / f'test_{stage}_stats.txt', 'a') as stats_file:
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
    def __init__(self, args, stage, stage1_backbone=None, stage1_projector=None, stage2_backbone=None, stage2_projector=None):
        super().__init__()
        self.args = args
        self.stage = stage

        if args.arch == 'resnet18':
            self.backbone1 = torchvision.models.resnet18(zero_init_residual=True)
            self.backbone1.fc = nn.Identity()
        elif args.arch == 'resnet50':
            self.backbone1 = torchvision.models.resnet50(zero_init_residual=True)
            self.backbone1.fc = nn.Identity()
        elif args.arch == 'mlp':
            self.backbone1 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28*3, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 512),
            )

        # Set models from stage 1 and freeze
        if stage == 'stage2' or stage == 'stage3':
            self.backbone2 = stage1_backbone
            self.projector2 = stage1_projector

            self.backbone2.eval()
            self.projector2.eval()

            for p in chain(self.backbone2.parameters(), self.projector2.parameters()):
                p.requires_grad = False
        
        if stage == 'stage3':
            self.backbone3 = stage2_backbone
            self.projector3 = stage2_projector

            self.backbone3.eval()
            self.projector3.eval()

            for p in chain(self.backbone3.parameters(), self.projector3.parameters()):
                p.requires_grad = False


        # projector
        sizes = [512] * self.args.layer + [self.args.dim]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[-1]))
        self.projector1 = nn.Sequential(*layers)

        self.onne_head_digit1 = nn.Linear(512, 10)
        self.onne_head_color1 = nn.Linear(512, 10)
        self.loss_fn = infoNCE_diverse

    def forward(self, y1, y2, digit_labels=None, color_labels=None, temp=0.1, lamb=1.0):
        r1_1 = self.backbone1(y1)
        r1_2 = self.backbone1(y2)

        # projection
        z1_1 = self.projector1(r1_1)
        z1_2 = self.projector1(r1_2)

        if self.stage == 'stage1':
            loss, reg_loss = self.loss_fn(z1_1, z1_2, temperature=temp)
        
        elif self.stage == 'stage2':
            r2_1 = self.backbone2(y1)
            r2_2 = self.backbone2(y2)

            z2_1 = self.projector2(r2_1)
            z2_2 = self.projector2(r2_2)

            loss, reg_loss = self.loss_fn(z1_1, z1_2, z2_1, z2_2, temperature=temp, stage=self.stage, lamb=lamb)
        
        elif self.stage == 'stage3':
            r2_1 = self.backbone2(y1)
            r2_2 = self.backbone2(y2)

            z2_1 = self.projector2(r2_1)
            z2_2 = self.projector2(r2_2)

            r3_1 = self.backbone3(y1)
            r3_2 = self.backbone3(y2)

            z3_1 = self.projector3(r3_1)
            z3_2 = self.projector3(r3_2)

            loss, reg_loss = self.loss_fn(z1_1, z1_2, z2_1, z2_2, z3_1, z3_2, temperature=temp, stage=self.stage, lamb=lamb)    

        # Online classifier 
        logits_digit1 = self.onne_head_digit1(r1_1.detach())
        logits_color1 = self.onne_head_color1(r1_1.detach())

        cls_digit_loss1 = torch.nn.functional.cross_entropy(logits_digit1, digit_labels)
        cls_color_loss1 = torch.nn.functional.cross_entropy(logits_color1, color_labels)

        digit_acc1 = torch.sum(torch.eq(torch.argmax(logits_digit1, dim=1), digit_labels)) / logits_digit1.size(0)
        color_acc1 = torch.sum(torch.eq(torch.argmax(logits_color1, dim=1), color_labels)) / logits_color1.size(0)

        clf_loss = cls_digit_loss1 + cls_color_loss1

        return loss, reg_loss, clf_loss, digit_acc1, color_acc1


def infoNCE(z1, z2, temperature=0.1):
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    z1 = gather_from_all(z1)
    z2 = gather_from_all(z2)
    logits = z1 @ z2.T
    logits /= temperature
    n = z2.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

def infoNCE_diverse(z1_1, z1_2, z2_1=None, z2_2=None, z3_1=None, z3_2=None, temperature=0.1, stage='', lamb=1.0):
    z1_1 = torch.nn.functional.normalize(z1_1, dim=1)
    z1_2 = torch.nn.functional.normalize(z1_2, dim=1)
    z1_1 = gather_from_all(z1_1)
    z1_2 = gather_from_all(z1_2)

    z1 = torch.cat([z1_1, z1_2], dim=0)

    sim_matrix1 = z1 @ z1.T
    sim_matrix1 /= temperature

    bs = z1_1.shape[0]
    n_views = 2
    device = z1_1.device

    labels = torch.cat([torch.arange(bs) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    labels = torch.argmax(labels, dim=1)

    sim_matrix1 = sim_matrix1[~mask].view(sim_matrix1.shape[0], -1)
    loss = torch.nn.functional.cross_entropy(sim_matrix1, labels.long())
    
    reg_loss = torch.Tensor([0.0]).to(device)

    if stage == 'stage2':
        z2_1 = torch.nn.functional.normalize(z2_1, dim=1)
        z2_2 = torch.nn.functional.normalize(z2_2, dim=1)
        z2_1 = gather_from_all(z2_1)
        z2_2 = gather_from_all(z2_2)
        z2 = torch.cat([z2_1, z2_2], dim=0)
        sim_matrix2 = z2 @ z2.T
        sim_matrix2 /= temperature

        sim_matrix2 = sim_matrix2[~mask].view(sim_matrix2.shape[0], -1)

        reg_loss = -1.0 * lamb * torch.nn.functional.l1_loss(torch.exp(sim_matrix1), torch.exp(sim_matrix2))

    if stage == 'stage3':
        z2_1 = torch.nn.functional.normalize(z2_1, dim=1)
        z2_2 = torch.nn.functional.normalize(z2_2, dim=1)
        z2_1 = gather_from_all(z2_1)
        z2_2 = gather_from_all(z2_2)
        z2 = torch.cat([z2_1, z2_2], dim=0)
        sim_matrix2 = z2 @ z2.T
        sim_matrix2 /= temperature

        z3_1 = torch.nn.functional.normalize(z3_1, dim=1)
        z3_2 = torch.nn.functional.normalize(z3_2, dim=1)
        z3_1 = gather_from_all(z3_1)
        z3_2 = gather_from_all(z3_2)
        z3 = torch.cat([z3_1, z3_2], dim=0)
        sim_matrix3 = z3 @ z3.T
        sim_matrix3 /= temperature

        sim_matrix2 = sim_matrix2[~mask].view(sim_matrix2.shape[0], -1)
        sim_matrix3 = sim_matrix3[~mask].view(sim_matrix3.shape[0], -1)

        reg_loss = -1.0 * lamb * (torch.nn.functional.l1_loss(torch.exp(sim_matrix1), torch.exp(sim_matrix2)) 
                                + torch.nn.functional.l1_loss(torch.exp(sim_matrix1), torch.exp(sim_matrix3)))

    return loss, reg_loss

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

        self.transform_cmnist = transforms.Compose([
            transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            # ], p=0.8),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        # std=[0.229, 0.224, 0.225])
        ])


    def __call__(self, x):
        y1 = self.transform_cmnist(x)
        y2 = self.transform_cmnist(x)
        return y1, y2