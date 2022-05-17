from __future__ import print_function

import os
import sys
import argparse
import time
import math

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss, simclr_loss, simclr_loss_pos_append, simclr_loss_pos_all
from losses import essl_loss, essl_loss_pos_append

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
import numpy as np
from pathlib import Path
from util import print_args, setup_wandb, logging_file, get_last_checkpoint
from util_knn import knn_loop
import pdb
st = pdb.set_trace


def set_test_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if opt.dataset == 'cifar10':
        memory_dataset = datasets.CIFAR10(root=opt.data_folder, train=True,
                                         transform=normalize,
                                         download=True)
        test_dataset = datasets.CIFAR10(root=opt.data_folder, train=False,
                                         transform=normalize,
                                         download=True)
    elif opt.dataset == 'cifar100':
        memory_dataset = datasets.CIFAR100(root=opt.data_folder, train=True,
                                          transform=normalize,
                                          download=True)
        test_dataset = datasets.CIFAR100(root=opt.data_folder, train=False,
                                          transform=normalize,
                                          download=True)
    elif opt.dataset == 'path' or opt.dataset == 'imagenet':
        memory_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'train'),
                                            transform=normalize)
        test_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'val_imagefolder'),
                                            transform=normalize)  # NOTE: hardcoded for T64
    else:
        raise ValueError(opt.dataset)
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)
    return memory_loader, test_loader


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path' or opt.dataset == 'imagenet':
        train_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'train'),
                                            transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)
    
    if opt.append_view:
        from util_data import MultiViewDataset
        transform3 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            normalize,
        ])  # NOTE: hardcoded, also NO post-transform
        train_dataset = MultiViewDataset(
            orig_dataset=train_dataset,
            pos_view_paths=opt.pos_view_paths,
            neg_view_paths=opt.neg_view_paths,
            transform3=transform3,
            n_views=1,  # NOTE: hardcoded
            train=True,
            subset_index=np.arange(len(train_dataset)),  # TODO: hardcoded
            sample_from_original=opt.sample_from_original,
            uint8=opt.uint8,
        )

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt, file_to_update=None):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (inputs, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # images = torch.cat([inputs[0], inputs[1]], dim=0)
        images = torch.cat(inputs, dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        if opt.split:
            features = torch.cat([model(image_) for image_ in images.split(opt.batch_size, dim=0)], dim=0)
        else:
            features = model(images)  # NOTE: forward large batch including appended views

        if opt.method in ['SupCon', 'SimCLR']:
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        else:
            feat_list = torch.split(features, bsz, dim=0)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        elif opt.method == 'SimCLR2':
            loss = simclr_loss(feat_list[0], feat_list[1], opt.temp)
        elif opt.method == 'SimCLR2+pos_app':
            loss = simclr_loss_pos_append(feat_list[0], feat_list[1], feat_list[2], opt.alpha, opt.temp)
        elif opt.method == 'SimCLR2+pos_all':
            loss = simclr_loss_pos_all(feat_list[0], feat_list[1], feat_list[2], opt.temp)
        elif opt.method == 'essl':
            loss = essl_loss(feat_list[0], feat_list[1], opt.temp)
        elif opt.method == 'essl+pos_app':
            loss = essl_loss_pos_append(feat_list[0], feat_list[1], feat_list[2], opt.alpha, opt.temp)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            line_to_print = ('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            print(line_to_print)
            sys.stdout.flush()

            if file_to_update:
                file_to_update.write(line_to_print + '\n')
                file_to_update.flush()

    return losses.avg


def main(opt):
    # set random seed
    # if not opt.no_seed:
    #     from util import fix_seed
    #     fix_seed(opt.seed)

    try:
        file_to_update = logging_file(os.path.join(opt.log_dir, 'train.log.txt'), 'a+')
    except IOError:
        file_to_update = None

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = setup_wandb(opt) if opt.use_wandb else None

    # resume from checkpoint
    start_epoch = 1
    if opt.resume:
        # ckpt_path = opt.resume_from
        ckpt_path = get_last_checkpoint(
            ckpt_dir=os.path.join(opt.log_dir, 'weights'),
            ckpt_ext='.pth',
        )
        assert os.path.isfile(ckpt_path), '{} is not a valid file.'.format(ckpt_path)
        print('loading checkpoint {}'.format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"starting from epoch {start_epoch}")

    memory_loader, test_loader = set_test_loader(opt)

    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt, file_to_update)
        time2 = time.time()

        # test
        knn_acc = knn_loop(model.encoder, memory_loader, test_loader)
        line_to_print = 'epoch {}, knn_acc {:.2f}, total time {:.2f}'.format(epoch, knn_acc, time2 - time1)
        print(line_to_print)
        if file_to_update:
            file_to_update.write(line_to_print + '\n')
            file_to_update.flush()

        if logger is not None:
            # logger.log({
            #     'loss': loss,
            #     'knn_acc': knn_acc,
            #     'lr': optimizer.param_groups[0]['lr'],
            #     'epoch': epoch,
            # }, step=epoch)
            logger.add_scalar('loss', loss, epoch)
            logger.add_scalar('knn_acc', knn_acc, epoch)
            logger.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            logger.add_scalar('epoch', epoch, epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.log_dir, 'weights', 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.log_dir, 'weights', 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
