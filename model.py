
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch


class BasicModel(nn.Module):
    def __init__(self, model_config, dataloader ):
        super(BasicModel, self).__init__()
        self.model_fig = model_config
        self.name = model_config['name']
        self.device = model_config['device']
        self.dataset = model_config['dataset']
        self.batch_size = model_config['train_batch_size']
        self.num_users = model_config['num_users']
        self.num_items = model_config['num_items']
        self.n_users = dataloader.dataset.get_user_num()
        self.n_items = dataloader.dataset.get_item_num()

        self.v_feat, self.t_feat = None, None

    def abstract_calculate_loss(self, user_score, pos_item_score, neg_item_score):
        mf_loss = self.loss(pos_item_score, neg_item_score)
        reg_loss = self.reg_loss(user_score, pos_item_score, neg_item_score)
        return mf_loss, reg_loss

    def get_user_item_embeddings(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings
        return user_e, all_item_e

class VBPR(BasicModel):
    def __init__(self, model_config, dataloader):
        super(VBPR, self).__init__(model_config, dataloader)
        dataset_path = os.path.abspath(model_config['data_path'] + model_config['dataset'])
        # 构建视觉、文本特征文件的路径
        v_feat_file_path = os.path.join(dataset_path, model_config['vision_feature_file'])
        t_feat_file_path = os.path.join(dataset_path, model_config['text_feature_file'])
        # 加载视觉、文本特征
        if os.path.isfile(v_feat_file_path):
            self.v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                self.device)
        if os.path.isfile(t_feat_file_path):
            self.t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor).to(
                self.device)

        assert self.v_feat is not None or self.t_feat is not None, 'Features all NONE'

        self.u_embedding_size = self.i_embedding_size = model_config['embedding_size']
        self.reg_weight = model_config['reg_weight']  # float32 type: the weight decay for l2 normalizaton

        # define layers and loss
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.i_embedding_size)))
        if self.v_feat is not None and self.t_feat is not None:
            self.item_raw_features = torch.cat((self.t_feat, self.v_feat), -1)
        elif self.v_feat is not None:
            self.item_raw_features = self.v_feat
        else:
            self.item_raw_features = self.t_feat

        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)

    def forward(self, item_raw_features, dropout=0.0):
        item_embeddings = self.item_linear(item_raw_features)
        item_embeddings = torch.cat((self.i_embedding, item_embeddings), dim=1)
        user_e = F.dropout(self.u_embedding, dropout)
        item_e = F.dropout(item_embeddings, dropout)
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_embeddings, item_embeddings = self.forward()
        user_e = user_embeddings[user, :]
        pos_e = item_embeddings[pos_item, :]
        neg_e = item_embeddings[neg_item, :]
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)

        mf_loss, reg_loss = self.abstract_calculate_loss(user_e, pos_item_score, neg_item_score)

        loss = mf_loss + self.reg_weight * reg_loss
        return loss

    def full_sort_predict(self, interaction):
        user_e, all_item_e = self.get_user_item_embeddings(interaction)
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

class MMGCN(BasicModel):
    def __init__(self, model_config, dataset):
        super(MMGCN, self).__init__(model_config, dataset)
        self.num_user = self.n_users
        self.num_user = self.n_users
        self.num_item = self.n_items
        dim_x = model_config['embedding_size']
        num_layer = model_config['n_layers']
        self.aggr_mode = 'mean'
        self.concate = 'False'
        has_id = True
        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)
        self.reg_weight = model_config['reg_weight']

        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        self.num_modal = 0

        if self.v_feat is not None:
            self.v_gcn = GCN(self.edge_index, self.batch_size, self.num_user, self.num_item, self.v_feat.size(1), dim_x,
                             self.aggr_mode,
                             self.concate, num_layer=num_layer, has_id=has_id, dim_latent=256, device=self.device)
            self.num_modal += 1

        if self.t_feat is not None:
            self.t_gcn = GCN(self.edge_index, self.batch_size, self.num_user, self.num_item, self.t_feat.size(1), dim_x,
                             self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, device=self.device)
            self.num_modal += 1

        self.id_embedding = nn.init.xavier_normal_(
            torch.rand((self.num_user + self.num_item, dim_x), requires_grad=True)).to(self.device)
        self.result = nn.init.xavier_normal_(torch.rand((self.num_user + self.num_item, dim_x))).to(self.device)

    def forward(self):
        representation = None
        if self.v_feat is not None:
            representation = self.v_gcn(self.v_feat, self.id_embedding)
        if self.t_feat is not None:
            if representation is None:
                representation = self.t_gcn(self.t_feat, self.id_embedding)
            else:
                representation += self.t_gcn(self.t_feat, self.id_embedding)

        representation /= self.num_modal

        self.result = representation
        return representation

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def calculate_loss(self, interaction):
        batch_users = interaction[0]
        pos_items = interaction[1] + self.n_users
        neg_items = interaction[2] + self.n_users

        user_tensor = batch_users.repeat_interleave(2)
        stacked_items = torch.stack((pos_items, neg_items))
        item_tensor = stacked_items.t().contiguous().view(-1)

        out = self.forward()
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score * item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        reg_embedding_loss = (self.id_embedding[user_tensor] ** 2 + self.id_embedding[item_tensor] ** 2).mean()
        if self.v_feat is not None:
            reg_embedding_loss += (self.v_gcn.preference ** 2).mean()
        reg_loss = self.reg_weight * reg_embedding_loss
        return loss + reg_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result[:self.n_users]
        item_tensor = self.result[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

class Abstract_GCN(nn.Module):
    def __init__(self, in_channels, out_channels, aggr='mean'):
        super(Abstract_GCN, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.aggr = aggr

    def forward(self, x, edge_index):
        row, col = edge_index
        if self.aggr == 'mean':
            out = torch.zeros_like(x)
            out.index_add_(0, row, x[col])
            out /= torch.bincount(row).unsqueeze(-1).float()
        else:
            raise NotImplementedError(f"Aggregation method {self.aggr} not implemented")
        return self.linear(out)

class GCN(nn.Module):
    def __init__(self, edge_index, batch_size, num_user, num_item, dim_feat, dim_id, aggr_mode, concate, num_layer,
                 has_id, dim_latent=None, device='cpu'):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id
        self.device = device

        self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent or self.dim_feat), requires_grad=True)).to(self.device)
        self.MLP = nn.Linear(self.dim_feat, self.dim_latent) if self.dim_latent else None

        self.layers = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.g_layers = nn.ModuleList()

        for i in range(num_layer):
            dim_in = self.dim_latent if self.dim_latent and i == 0 else self.dim_id
            dim_out = self.dim_id
            self.layers.append(Abstract_GCN(dim_in, dim_out, aggr=self.aggr_mode))
            self.linears.append(nn.Linear(dim_in, dim_out))
            g_layer_in_dim = dim_out + dim_out if self.concate else dim_out
            self.g_layers.append(nn.Linear(g_layer_in_dim, dim_out))

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.linear.weight)
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)
        for g_layer in self.g_layers:
            nn.init.xavier_normal_(g_layer.weight)

    def forward(self, features, id_embedding):
        x = self.MLP(features) if self.MLP else features
        x = torch.cat((self.preference, x), dim=0)
        x = F.normalize(x)

        for i in range(self.num_layer):
            h = F.leaky_relu(self.layers[i](x, self.edge_index))
            x_hat = F.leaky_relu(self.linears[i](x)) + id_embedding if self.has_id else F.leaky_relu(self.linears[i](x))
            x = F.leaky_relu(self.g_layers[i](torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layers[i](h) + x_hat)

        return x