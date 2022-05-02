import os
import pickle
import numpy as np

import torch
import torchvision
import torchvision.transforms as T

import pdb
st = pdb.set_trace


class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        orig_dataset=None,
        latent_path='latent.pkl',
        pos_view_paths=[],
        neg_view_paths=[],
        transform3=None,
        n_views=1,
        train=True,
        subset_index=None,
        sample_from_original=False,
    ):
        """
        images: [x0, x1, x3, ...]
            x0, x1 are two expert views
            x3 and after are for (cached) GAN generated views
        returns: (images, labels)
        """
        self.orig_dataset = orig_dataset
        # NOTE: transform 0 and 1 are handled in the original dataset
        # self.transform0 = transform0
        # self.transform1 = transform1
        self.transform3 = transform3

        # load pos views
        if isinstance(pos_view_paths, str):
            pos_view_paths = [pos_view_paths]
        views = []
        for view_path in pos_view_paths:
            assert os.path.exists(view_path), f'pos_view_path {view_path} does not exist'
            with open(view_path, 'rb') as f:
                view = pickle.load(f)
            views.append(view['views'][:, subset_index])
        self.views_pos = torch.cat(views, dim=0) if len(views) > 0 else None
        self.total_views_pos = self.views_pos.shape[0] if len(views) > 0 else 0
        self.sample_pos = len(pos_view_paths) > 0

        # load neg views
        if isinstance(neg_view_paths, str):
            neg_view_paths = [neg_view_paths]
        views = []
        for view_path in neg_view_paths:
            assert os.path.exists(view_path), f'neg_view_path {view_path} does not exist'
            with open(view_path, 'rb') as f:
                view = pickle.load(f)
            views.append(view['views'][:, subset_index])
        self.views_neg = torch.cat(views, dim=0) if len(views) > 0 else None
        self.total_views_neg = self.views_neg.shape[0] if len(views) > 0 else 0
        self.sample_neg = len(neg_view_paths) > 0

        self.n_views = n_views  # NOTE: for each pos or neg
        if self.views_pos is not None:
            assert sample_from_original or self.n_views <= self.total_views_pos
        if self.views_neg is not None:
            assert sample_from_original or self.n_views <= self.total_views_neg
        assert len(subset_index) == len(orig_dataset)
        self.dataset_len = len(orig_dataset)
        self.sample_original = sample_from_original
        assert not sample_from_original  # NOTE: needs to be handled differently

    def data_transform(self, image_tensor):
        image_tensor = (image_tensor + 1) / 2
        return image_tensor

    def __getitem__(self, index):
        images, labels = self.orig_dataset[index]

        # view_3 and ...
        views = []
        if self.sample_pos:
            inds_pos = list(np.random.choice(np.arange(self.total_views_pos), size=self.n_views, replace=False))
            for i in inds_pos:
                img = self.data_transform(self.views_pos[i, index])
                if self.transform3 is not None:
                    img = self.transform3(img)
                views.append(img)
            # views += [self.transform3(self.data_transform(self.views_pos[i, index])) for i in inds_pos]
        if self.sample_neg:
            inds_neg = list(np.random.choice(np.arange(self.total_views_neg), size=self.n_views, replace=False))
            for i in inds_neg:
                img = self.data_transform(self.views_neg[i, index])
                if self.transform3 is not None:
                    img = self.transform3(img)
                views.append(img)
        images += views
        return images, labels

    def __len__(self):
        return self.dataset_len