"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from dist_utils import gather_from_all
import pdb
st = pdb.set_trace

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
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

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

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
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class SupConLoss1(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss1, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features1, features2, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features1.is_cuda
                  else torch.device('cpu'))

        features1 = gather_from_all(features1)
        features2 = gather_from_all(features2)
        labels = gather_from_all(labels)

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
        if self.contrast_mode == 'one':
            anchor_feature = features1
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

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
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def simclr_loss(z1, z2, temperature=0.07, base_temperature=0.07):
    """
    reimplement simclr loss see if it works
    """
    bsz = z1.shape[0]
    bsz2 = bsz * 2

    labels = torch.cat([torch.arange(bsz) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().cuda()

    z_anchor = torch.cat([z1, z2], dim=0)
    z_contra = z_anchor

    logits = z_anchor @ z_contra.T
    mask = torch.eye(bsz2, dtype=torch.bool).cuda()
    labels = labels[~mask].view(bsz2, -1)
    logits = logits[~mask].view(bsz2, -1)
    positives = logits[labels.bool()].view(bsz2, -1)
    negatives = logits[~labels.bool()].view(bsz2, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    label = torch.zeros(bsz2, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, label)
    loss = loss * (temperature / base_temperature)

    return loss

def simclr_loss_pos_all(z1, z2, z3, temperature=0.07, base_temperature=0.07):
    """
    full simclr loss, append positive by supcon loss
    """
    bsz = z1.shape[0]
    bsz3 = bsz * 3

    labels = torch.cat([torch.arange(bsz) for i in range(3)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().cuda()

    z_anchor = torch.cat([z1, z2, z3], dim=0)
    z_contra = z_anchor

    logits = z_anchor @ z_contra.T
    mask = torch.eye(bsz3, dtype=torch.bool).cuda()
    labels = labels[~mask].view(bsz3, -1)
    logits = logits[~mask].view(bsz3, -1)
    positives = logits[labels.bool()].view(bsz3, -1)
    negatives = logits[~labels.bool()].view(bsz3, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    label0 = torch.zeros(bsz3, dtype=torch.long).cuda()
    loss0 = torch.nn.functional.cross_entropy(logits, label0)
    label1 = torch.ones(bsz3, dtype=torch.long).cuda()
    loss1 = torch.nn.functional.cross_entropy(logits, label1)
    loss = (loss0 + loss1) / 2 * (temperature / base_temperature)

    return loss

def simclr_loss_pos_append(z1, z2, z3, alpha=0.5, temperature=0.07, base_temperature=0.07):
    """
    full simclr loss, append inner product
    """
    bsz = z1.shape[0]
    bsz2 = bsz * 2

    labels = torch.cat([torch.arange(bsz) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().cuda()

    z_anchor = torch.cat([z1, z2], dim=0)
    z_contra = z_anchor

    logits = z_anchor @ z_contra.T
    mask = torch.eye(bsz2, dtype=torch.bool).cuda()
    labels = labels[~mask].view(bsz2, -1)
    logits = logits[~mask].view(bsz2, -1)
    positives = logits[labels.bool()].view(bsz2, -1)
    negatives = logits[~labels.bool()].view(bsz2, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    label = torch.zeros(bsz2, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, label)

    inner_prod = torch.mean(torch.sum(z1 * z3, dim=1)) / 2 + torch.mean(torch.sum(z2 * z3, dim=1)) / 2
    loss = loss - alpha * inner_prod / temperature

    loss = loss * (temperature / base_temperature)

    return loss


def info_nce_loss(z1, z2, temperature=0.5):
    # z1 = torch.nn.functional.normalize(z1, dim=1)
    # z2 = torch.nn.functional.normalize(z2, dim=1)

    logits = z1 @ z2.T
    logits /= temperature
    n = z2.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


def essl_loss(z1, z2, temperature=0.07, base_temperature=0.07):
    """
    essl simclr loss
    """
    loss = info_nce_loss(z1, z2, temperature) / 2 + info_nce_loss(z2, z1, temperature) / 2
    loss = loss * (temperature / base_temperature)
    return loss


def essl_loss_pos_append(z1, z2, z3, alpha=0.5, temperature=0.07, base_temperature=0.07):
    """
    essl simclr loss, append inner product
    """
    loss = info_nce_loss(z1, z2, temperature) / 2 + info_nce_loss(z2, z1, temperature) / 2
    inner_prod = torch.mean(torch.sum(z1 * z3, dim=1)) / 2 + torch.mean(torch.sum(z2 * z3, dim=1)) / 2
    loss = loss - alpha * inner_prod / temperature
    
    loss = loss * (temperature / base_temperature)

    return loss


"""
TODO: clean this up later
"""
class SimCLRLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
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

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            raise ValueError('`contrast_mode` cannot be `one`')
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        elif self.contrast_mode == 'append':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

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
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss