import numpy as np
import torch
import pandas as pd
import random
from torch.utils.data import Dataset, random_split


def update_ui_sets(u, i, user_inter_sets, item_inter_sets):
    if u in user_inter_sets:
        user_inter_sets[u].add(i)
    else:
        user_inter_sets[u] = {i}
    if i in item_inter_sets:
        item_inter_sets[i].add(u)
    else:
        item_inter_sets[i] = {u}


#
def update_user_inter_lists(u, i, t, user_map, item_map, user_inter_lists):
    if u in user_map and i in item_map:
        duplicate = False
        for i_t in user_inter_lists[user_map[u]]:
            if i_t[0] == item_map[i]:
                i_t[1] = min(i_t[1], t)
                duplicate = True
                break
        if not duplicate:
            user_inter_lists[user_map[u]].append([item_map[i], t])


def reindex_feat(feat, item_map):
    indexes = []
    for idx in range(feat.shape[0]):
        if idx in item_map:
            indexes.append(item_map[idx])
    indexes = torch.tensor(indexes, dtype=torch.long, device=feat.device)
    feat = torch.index_select(feat, dim=0, index=indexes)
    return feat


class AmazonDataset(Dataset):
    def __init__(self, dataset_config):
        print(dataset_config)
        self.name = dataset_config['name']
        self.device = dataset_config['device']
        self.split_ratio = dataset_config['split_ratio']
        self.shuffle = dataset_config.get('shuffle', False)
        self.min_interactions = dataset_config['min_inter']

        self.v_feat = self.process_feat(dataset_config.get('image_feat_path', None))
        self.t_feat = self.process_feat(dataset_config.get('text_feat_path', None))
        self.n_users = self.n_items = None

        item_map, user_inter_lists = self.read_interactions(dataset_config['interactions_path'])
        self.v_feat = reindex_feat(self.v_feat, item_map)
        self.t_feat = reindex_feat(self.t_feat, item_map)

        self.train_data = self.val_data = self.train_array = None
        self.train_array = []

        self.generate_data(user_inter_lists)

    def process_feat(self, feat_path):
        if feat_path is None:
            return None
        feat = torch.tensor(np.load(feat_path), dtype=torch.float32, device=self.device)
        return feat

    def remove_sparse_ui(self, user_inter_sets, item_inter_sets):
        not_stop = True
        while not_stop:
            not_stop = False
            users = list(user_inter_sets.keys())
            for user in users:
                if len(user_inter_sets[user]) < self.min_interactions:
                    not_stop = True
                    for item in user_inter_sets[user]:
                        item_inter_sets[item].remove(user)
                    user_inter_sets.pop(user)
            items = list(item_inter_sets.keys())
            for item in items:
                if len(item_inter_sets[item]) < self.min_interactions:
                    not_stop = True
                    for user in item_inter_sets[item]:
                        user_inter_sets[user].remove(item)
                    item_inter_sets.pop(item)
        user_map = dict()
        for idx, user in enumerate(user_inter_sets):
            user_map[user] = idx
        item_map = dict()
        for idx, item in enumerate(item_inter_sets):
            item_map[item] = idx
        self.n_users = len(user_map)
        self.n_items = len(item_map)
        return user_map, item_map

    def read_interactions(self, interactions_path):
        interactions = pd.read_csv(interactions_path, sep='\t')

        user_inter_sets, item_inter_sets = dict(), dict()
        for u, i, r, t, _ in interactions.itertuples(index=False):
            if r > 3:
                update_ui_sets(u, i, user_inter_sets, item_inter_sets)
        user_map, item_map = self.remove_sparse_ui(user_inter_sets, item_inter_sets)

        user_inter_lists = [[] for _ in range(self.n_users)]
        for u, i, r, t, _ in interactions.itertuples(index=False):
            if r > 3:
                update_user_inter_lists(u, i, t, user_map, item_map, user_inter_lists)
        return item_map, user_inter_lists

    def generate_data(self, user_inter_lists):
        self.train_data = [None for _ in range(self.n_users)]
        self.val_data = [None for _ in range(self.n_users)]
        average_inters = []
        for user in range(self.n_users):
            user_inter_lists[user].sort(key=lambda entry: entry[1])
            if self.shuffle:
                np.random.shuffle(user_inter_lists[user])

            n_inter_items = len(user_inter_lists[user])
            average_inters.append(n_inter_items)
            n_train_items = int(n_inter_items * self.split_ratio)
            self.train_data[user] = {i_t[0] for i_t in user_inter_lists[user][:n_train_items]}
            self.val_data[user] = {i_t[0] for i_t in user_inter_lists[user][n_train_items:]}

            for item in self.train_data[user]:
                self.train_array.append([user, item])

        average_inters = np.mean(average_inters)
        print('Users {:d}, Items {:d}, Average number of interactions {:.3f}, Total interactions {:.1f}'
              .format(self.n_users, self.n_items, average_inters, average_inters * self.n_users))

    def get_negative_item(self, user):
        pos_items = self.train_data[user]
        item = random.randint(0, self.n_items - 1)
        while item in pos_items:
            item = random.randint(0, self.n_items - 1)
        return item

    def __len__(self):
        return len(self.train_array)

    def __getitem__(self, index):
        user = random.randint(0, self.n_users - 1)
        while len(self.train_data[user]) == 0:
            user = random.randint(0, self.n_users - 1)

        pos_item = np.random.choice(list(self.train_data[user]))
        neg_item = self.get_negative_item(user)
        return user, pos_item, neg_item




