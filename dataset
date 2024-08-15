import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import random

class BaseDataset(Dataset):
    def __init__(self, dataset_config, interactions):
        print(dataset_config)
        self.name = dataset_config['name']
        self.device = dataset_config['device']
        self.split_ratio = dataset_config.get('split_ratio')
        self.negative_sample_ratio = dataset_config.get('negative_sample_ratio', 1)
        self.shuffle = dataset_config.get('shuffle', False)
        self.min_interactions = dataset_config.get('min_inter')
        self.interactions = interactions
        self.n_users = np.max(interactions[:, 0]) + 1
        self.n_items = np.max(interactions[:, 1]) + 1
        self.train_data = self._generate_train_data()
        self.val_data = {}

    def _generate_train_data(self):
        train_data = {}
        for (user, item) in self.interactions:
            if user not in train_data:
                train_data[user] = set()
            train_data[user].add(item)
        return train_data

    def get_negative_samples(self, user, num_samples):
        positive_items = self.train_data.get(user, set())
        negative_items = []
        while len(negative_items) < num_samples:
            item = random.randint(0, self.n_items - 1)
            if item not in positive_items:
                negative_items.append(item)
        return negative_items

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, index):
        user, pos_item = self.interactions[index]
        neg_items = self.get_negative_samples(user, self.negative_sample_ratio)
        return {
            'user': torch.tensor(user, dtype=torch.long),
            'pos_item': torch.tensor(pos_item, dtype=torch.long),
            'neg_items': torch.tensor(neg_items, dtype=torch.long)
        }

    def split(self):
        train_size = int(len(self.interactions) * self.split_ratio)
        val_size = len(self.interactions) - train_size
        train_set, val_set = random_split(self, [train_size, val_size])
        return train_set, val_set

