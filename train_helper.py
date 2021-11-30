import logging
import os
import pickle
import time
import math
import random
import json
import copy

import torch
from torch import nn, optim
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np


from dataset import ColoredMNIST
from dist_utils import gather_from_all, init_distributed_mode, get_rank, is_main_process, get_world_size

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

    if is_main_process():
        try:
            tb_logger = SummaryWriter(args.log_dir + args.name)
        except:
            tb_logger = SummaryWriter(args.log_dir + args.name + '1')

    num_tasks = get_world_size()
    global_rank = get_rank()

    _logger.info('Creating dataset')
    dataset = ColoredMNIST(args.data, env='all_train', transform=Transform(args))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=False)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    sampler.set_epoch(0)

    _logger.info('Creating model')
    model = SimCLR(args).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    # optimizer = LARS(model.parameters(), lr=0, weight_decay=args.weight_decay,
    #                  weight_decay_filter=exclude_bias_and_norm,
    #                  lars_adaptation_filter=exclude_bias_and_norm)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

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
    itr = start_epoch * len(loader)

    _logger.info('Starting training')
    for epoch in range(start_epoch, args.epochs):
        _logger.info(f'Starting training epoch {epoch}')
        sampler.set_epoch(epoch)
        
        for step, ((y1, y2), digit_labels, color_labels) in enumerate(loader, start=epoch * len(loader)):
            itr_start = time.time()
            itr += 1
            y1 = y1.to(device, non_blocking=True)
            y2 = y2.to(device, non_blocking=True)

            lr = adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, digit_acc1, digit_acc2, color_acc1, color_acc2 = model.forward(y1, y2, digit_labels, color_labels)
                # loss, acc = model.forward(y1, y2, neg_images=neg_y, labels=labels, neg_labels=neg_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            itr_end = time.time()
            itr_time = itr_end - itr_start
            
            if is_main_process():
                tb_logger.add_scalar('loss', loss.item(), itr)
                tb_logger.add_scalar('acc/f_digit_acc', digit_acc1.item(), itr)
                tb_logger.add_scalar('acc/g_digit_acc', digit_acc2.item(), itr)
                tb_logger.add_scalar('acc/f_color_acc', color_acc1.item(), itr)
                tb_logger.add_scalar('acc/g_color_acc', color_acc2.item(), itr)

            if step % args.print_freq == 0:
                torch.distributed.reduce(digit_acc1.div_(args.world_size), 0)
                torch.distributed.reduce(digit_acc2.div_(args.world_size), 0)
                torch.distributed.reduce(color_acc1.div_(args.world_size), 0)
                torch.distributed.reduce(color_acc2.div_(args.world_size), 0)
                if is_main_process():
                    _logger.info(f'epoch={epoch}, step={step}, loss={loss.item()}, digit acc1={digit_acc1.item()}, digit acc1={digit_acc2.item()}, color acc1={color_acc1.item()}, color acc2={color_acc2.item()} itr time={itr_time}')
                    stats = dict(epoch=epoch, step=step, learning_rate=lr,
                                 loss=loss.item(), digit_acc1=digit_acc1.item(), digit_acc2=digit_acc2.item(),
                                 color_acc1=color_acc1.item(), color_acc2=color_acc2.item(),
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
                    args.checkpoint_dir + args.name + '-resnet18.pth')

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
        self.backbone1 = torchvision.models.resnet18(zero_init_residual=True)
        self.backbone1.fc = nn.Identity()

        self.backbone2 = torchvision.models.resnet18(zero_init_residual=True)
        self.backbone2.fc = nn.Identity()

        # projector
        sizes = [512] * self.args.layer + [self.args.dim]
        # sizes = [2048, 2048, 2048, self.args.dim]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        layers.append(nn.BatchNorm1d(sizes[-1]))
        self.projector1 = nn.Sequential(*layers)
        self.projector2 = nn.Sequential(*copy.deepcopy(layers))

        self.onne_head_digit1 = nn.Linear(512, 10)
        self.onne_head_digit2 = nn.Linear(512, 10)
        self.onne_head_color1 = nn.Linear(512, 1)
        self.onne_head_color2 = nn.Linear(512, 1)
        self.loss_fn = infoNCE_diverse

    def forward(self, y1, y2, digit_labels=None, color_labels=None):
        r1_1 = self.backbone1(y1)
        r1_2 = self.backbone1(y2)

        r2_1 = self.backbone2(y1)
        r2_2 = self.backbone2(y2)

        # projection
        z1_1 = self.projector1(r1_1)
        z1_2 = self.projector1(r1_2)

        z2_1 = self.projector2(r2_1)
        z2_2 = self.projector2(r2_2)

        loss = self.loss_fn(z1_1, z1_2, z2_1, z2_2)

        # Online classifier 
        logits_digit1 = self.onne_head_digit1(r1_1.detach())
        logits_color1 = self.onne_head_color1(r1_1.detach())

        logits_digit2 = self.onne_head_digit2(r2_1.detach())
        logits_color2 = self.onne_head_color2(r2_1.detach())

        cls_digit_loss1 = torch.nn.functional.cross_entropy(logits_digit1, digit_labels)
        cls_digit_loss2 = torch.nn.functional.cross_entropy(logits_digit2, digit_labels)

        cls_color_loss1 = torch.nn.functional.binary_cross_entropy_with_logits(logits_color1.squeeze(), color_labels.float())
        cls_color_loss2 = torch.nn.functional.binary_cross_entropy_with_logits(logits_color2.squeeze(), color_labels.float())

        digit_acc1 = torch.sum(torch.eq(torch.argmax(logits_digit1, dim=1), digit_labels)) / logits_digit1.size(0)
        digit_acc2 = torch.sum(torch.eq(torch.argmax(logits_digit2, dim=1), digit_labels)) / logits_digit2.size(0)

        logits_color1 = torch.nn.functional.sigmoid(logits_color1) > 0.5
        logits_color2 = torch.nn.functional.sigmoid(logits_color2) > 0.5

        color_acc1 = (logits_color1 == color_labels).sum() / logits_color1.size(0)
        color_acc2 = (logits_color2 == color_labels).sum() / logits_color2.size(0)

        loss += cls_digit_loss1 + cls_digit_loss2 + cls_color_loss1 + cls_color_loss2

        return loss, digit_acc1, digit_acc2, color_acc1, color_acc2


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

def infoNCE_diverse(z1_1, z1_2, z2_1, z2_2, temperature=0.1, lamb=1.0):
    z1_1 = torch.nn.functional.normalize(z1_1, dim=1)
    z1_2 = torch.nn.functional.normalize(z1_2, dim=1)
    z2_1 = torch.nn.functional.normalize(z2_1, dim=1)
    z2_2 = torch.nn.functional.normalize(z2_2, dim=1)

    z1_1 = gather_from_all(z1_1)
    z1_2 = gather_from_all(z1_2)
    z2_1 = gather_from_all(z2_1)
    z2_2 = gather_from_all(z2_2)

    z1 = torch.cat([z1_1, z1_2], dim=0)
    z2 = torch.cat([z2_1, z2_2], dim=0)

    sim_matrix1 = z1 @ z1.T
    sim_matrix1 /= temperature

    sim_matrix2 = z2 @ z2.T
    sim_matrix2 /= temperature

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
    sim_matrix2 = sim_matrix2[~mask].view(sim_matrix2.shape[0], -1)

    loss = torch.nn.functional.cross_entropy(sim_matrix1, labels.long()) + torch.nn.functional.cross_entropy(sim_matrix2, labels.long())
    loss += -1.0 * lamb * torch.nn.functional.l1_loss(torch.exp(sim_matrix1), torch.exp(sim_matrix2))

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

        self.transform_cmnist = transforms.Compose([
            transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        # std=[0.229, 0.224, 0.225])
        ])


    def __call__(self, x):
        y1 = self.transform_cmnist(x)
        y2 = self.transform_cmnist(x)
        return y1, y2