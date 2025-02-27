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

from util import TwoCropTransform, TwoCropTransform1, TwoCropTransform2, TwoCropTransform3, AverageMeter
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

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

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
            'v1=v2=gan+basic', 'v1=v2=basic+gan', 'v1=v2=basic', 'v3=basic,v1=v2=gan', 'v1=basic,v2=v3=gan',
            'v1=v2=v3=expert', 'v1=v2=v3=basic', 'v1=v2=v3=gan'])

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

    print_args(parser, opt)

    return opt


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
    basic_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    if opt.add_randomcrop:
        gan_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.9, 1.), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            normalize,
        ])
    elif opt.add_randomcrop2:
        gan_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.7, 1.)),
            transforms.RandomHorizontalFlip(),
            normalize,
        ])
    elif opt.add_randomcrop3:
        gan_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.5, 1.)),
            transforms.RandomHorizontalFlip(),
            normalize,
        ])
    else:
        gan_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            normalize,
        ])

    n_views_gan = 0
    if opt.setting == 'default':
        tt1 = TwoCropTransform(train_transform)
        tt3 = None
    elif opt.setting == 'v1=v2=gan':
        tt1 = None
        tt3 = gan_transform
        n_views_gan = 2
    elif opt.setting == 'v1=v2=basic':
        tt1 = TwoCropTransform(basic_transform)
        tt3 = None
    elif opt.setting == 'v1=basic,v2=gan':
        tt1 = TwoCropTransform1(basic_transform)
        tt3 = gan_transform
        n_views_gan = 1
    elif opt.setting == 'v1=expert,v2=gan':
        tt1 = TwoCropTransform1(train_transform)
        tt3 = gan_transform
        n_views_gan = 1
    elif opt.setting == 'v1=v2=v3=expert':
        from util import ThreeCropTransform
        # tt1 = TwoCropTransform3(train_transform,train_transform,train_transform)
        tt1 = ThreeCropTransform(train_transform)
        tt3 = None
        n_views_gan = 0
    elif opt.setting == 'v1=v2=v3=basic':
        from util import ThreeCropTransform
        # tt1 = TwoCropTransform3(train_transform,train_transform,train_transform)
        tt1 = ThreeCropTransform(basic_transform)
        tt3 = None
        n_views_gan = 0
    elif opt.setting == 'v1=v2=v3=gan':
        from util import ThreeCropTransform
        # tt1 = TwoCropTransform3(train_transform,train_transform,train_transform)
        tt1 = None
        tt3 = gan_transform
        n_views_gan = 3
    elif opt.setting == 'v1=v2=basic,v3=gan':
        tt1 = TwoCropTransform(basic_transform)
        tt3 = gan_transform
        n_views_gan = 1
    elif opt.setting == 'v3=basic,v1=v2=gan':
        tt1 = TwoCropTransform1(basic_transform)
        tt3 = gan_transform
        n_views_gan = 2
    elif opt.setting == 'v1=basic,v2=v3=gan':
        tt1 = TwoCropTransform1(basic_transform)
        tt3 = gan_transform
        n_views_gan = 2
    elif opt.setting == 'v1=v2=gan+basic':
        tt1 = None
        tt3 = transforms.Compose([
            transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            normalize,
        ])
        n_views_gan = 2
    else:
        raise ValueError('setting not supported: {}'.format(opt.setting))

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=tt1,
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=tt1,
                                          download=True)
    elif opt.dataset == 'path' or opt.dataset == 'imagenet':
        train_dataset = datasets.ImageFolder(root=os.path.join(opt.data_folder, 'train'),
                                            transform=tt1)
    else:
        raise ValueError(opt.dataset)

    from util_data import MultiViewDataset4
    train_dataset = MultiViewDataset4(
        orig_dataset=train_dataset,
        pos_view_paths=opt.pos_view_paths,
        neg_view_paths=opt.neg_view_paths,
        transform1=tt1,
        transform3=tt3,
        n_views=n_views_gan,
        train=True,
        subset_index=np.arange(len(train_dataset)),  # TODO: hardcoded
        sample_from_original=opt.sample_from_original,
        uint8=opt.uint8,
        setting=opt.setting,
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

        feat_list = torch.split(features, bsz, dim=0)

        if opt.setting == 'default':
            loss = simclr_loss(feat_list[0], feat_list[1], opt.temp)
        elif opt.setting == 'v1=v2=gan':
            loss = simclr_loss(feat_list[0], feat_list[1], opt.temp)
        elif opt.setting == 'v1=v2=basic':
            loss = simclr_loss(feat_list[0], feat_list[1], opt.temp)
        elif opt.setting == 'v1=basic,v2=gan':
            loss = simclr_loss(feat_list[0], feat_list[1], opt.temp)
        elif opt.setting == 'v1=expert,v2=gan':
            loss = simclr_loss(feat_list[0], feat_list[1], opt.temp)
        elif opt.setting == 'v1=v2=v3=expert' or opt.setting == 'v1=v2=v3=basic' or opt.setting == 'v1=v2=v3=gan':
            if opt.method == 'essl+diag':
                loss = essl_loss_pos_append(feat_list[0], feat_list[1], feat_list[2], opt.alpha, opt.temp)
            elif opt.method == 'simclr+all':
                loss = simclr_loss_pos_all(feat_list[0], feat_list[1], feat_list[2], opt.temp)
            else:
                raise ValueError('method not supported: {}'.format(opt.method))
        elif opt.setting == 'v1=v2=basic,v3=gan':
            if opt.method == 'essl+diag':
                loss = essl_loss_pos_append(feat_list[0], feat_list[1], feat_list[2], opt.alpha, opt.temp)
            elif opt.method == 'simclr+diag':
                loss = simclr_loss_pos_append(feat_list[0], feat_list[1], feat_list[2], opt.alpha, opt.temp)
            elif opt.method == 'simclr+all':
                loss = simclr_loss_pos_all(feat_list[0], feat_list[1], feat_list[2], opt.temp)
            else:
                raise ValueError('method not supported: {}'.format(opt.method))
        elif opt.setting == 'v3=basic,v1=v2=gan':
            if opt.method == 'essl+diag':
                loss = essl_loss_pos_append(feat_list[1], feat_list[2], feat_list[0], opt.alpha, opt.temp)
            elif opt.method == 'simclr+all':
                loss = simclr_loss_pos_all(feat_list[1], feat_list[2], feat_list[0], opt.temp)
            else:
                raise ValueError('method not supported: {}'.format(opt.method))
        elif opt.setting == 'v1=basic,v2=v3=gan':
            loss = essl_loss_pos_append(feat_list[0], feat_list[1], feat_list[2], opt.alpha, opt.temp)
        elif opt.setting == 'v1=v2=gan+basic':
            loss = simclr_loss(feat_list[0], feat_list[1], opt.temp)
        else:
            raise ValueError('setting not supported: {}'.format(opt.setting))

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


def main():
    opt = parse_option()

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
            logger.log({
                'loss': loss,
                'knn_acc': knn_acc,
                'lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
            }, step=epoch)

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
