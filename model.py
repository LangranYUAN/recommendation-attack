import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class BasicModel(nn.Module):
    def __init__(self, model_config):
        super(BasicModel, self).__init__()
        self.name = model_config['name']
        self.device = model_config['device']
        self.dataset = model_config['dataset']
        self.n_users = self.dataset.n_users
        self.n_items = self.dataset.n_items
        print(f"Number of users: {self.n_users}, Number of items: {self.n_items}")
        self.init_parameters()

    def init_parameters(self):
        raise NotImplementedError("Subclasses should initialize model parameters.")

    def get_rep(self):
        raise NotImplementedError

    def bpr_forward(self, users, pos_items, neg_items):
        rep = self.get_rep()
        user_rep = rep[users, :]
        pos_item_rep = rep[self.n_users + pos_items, :]
        neg_item_rep = rep[self.n_users + neg_items, :]

        l2_norm_sq = torch.norm(user_rep, p=2, dim=1) ** 2 + \
                     torch.norm(pos_item_rep, p=2, dim=1) ** 2 + \
                     torch.norm(neg_item_rep, p=2, dim=1) ** 2

        if self.v_feat is not None and hasattr(self, 'v_gcn'):
            l2_norm_sq += torch.norm(self.v_gcn.preference[users], p=2) ** 2

        return user_rep, pos_item_rep, neg_item_rep, l2_norm_sq

    def predict(self, users):
        rep = self.get_rep()
        user_rep = rep[users, :]
        all_items_rep = rep[self.n_users:, :]
        scores = torch.mm(user_rep, all_items_rep.t())
        return scores


class VBPR(BasicModel):
    def __init__(self, model_config):
        super(VBPR, self).__init__(model_config)
        self.v_feat = model_config['v_feat']
        self.t_feat = model_config['t_feat']
        self.dropout = model_config.get('dropout', 0.)

        self.embedding_size = model_config['embedding_size']

        if self.v_feat is not None and self.t_feat is not None:
            self.item_raw_features = torch.cat((self.t_feat, self.v_feat), -1)
        elif self.v_feat is not None:
            self.item_raw_features = self.v_feat
        elif self.t_feat is not None:
            self.item_raw_features = self.t_feat
        else:
            self.item_raw_features = None  # 当没有特征时的处理

        self.u_embedding = nn.Parameter(torch.empty(self.n_users, self.embedding_size * 2))
        self.i_embedding = nn.Parameter(torch.empty(self.n_items, self.embedding_size))

        # 物品线性变换层
        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.embedding_size)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.u_embedding)
        nn.init.xavier_normal_(self.i_embedding)
        nn.init.xavier_normal_(self.item_linear.weight)
        if self.item_linear.bias is not None:
            nn.init.zeros_(self.item_linear.bias)

    def get_rep(self):
        user_rep = F.dropout(self.u_embedding, self.dropout)
        item_feat_rep = self.item_linear(self.item_raw_features)
        item_emb_rep = F.dropout(self.i_embedding, self.dropout)

        # 合并用户表示和物品表示
        item_rep = torch.cat((item_emb_rep, item_feat_rep), dim=1)
        return torch.cat((user_rep, item_rep), dim=0)


class MMGCN(BasicModel):
    def __init__(self, model_config):
        super(MMGCN, self).__init__(model_config)
        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)

        train_interactions = model_config['dataset'].inter_matrix(form='coo').astype(np.float32)

        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0], :]), dim=1)
        self.num_modal = 0

        gcn_config = {
            'edge_index': self.edge_index,
            'num_user': self.n_users,
            'num_item': self.n_items,
            'dim_input': self.v_feat.size(1) if self.v_feat is not None else None,
            'dim_output': self.dim_x,
            'aggr_mode': self.aggr_mode,
            'concate': self.concate,
            'num_layer': self.num_layer
        }
        if self.v_feat is not None:
            self.v_gcn = GCN(gcn_config)
            self.num_modal += 1

        if self.t_feat is not None:
            self.t_gcn = GCN(gcn_config)
            self.num_modal += 1

        self.id_embedding = nn.init.xavier_normal_(
            torch.rand((self.num_user + self.num_item, self.dim_x), requires_grad=True)).to(self.device)

    def get_rep(self):
        representation = None
        if self.v_feat is not None:
            representation = self.v_gcn(self.v_feat, self.id_embedding)
        if self.t_feat is not None:
            if representation is None:
                representation = self.t_gcn(self.t_feat, self.id_embedding)
            else:
                representation += self.t_gcn(self.t_feat, self.id_embedding)

        representation /= self.num_modal
        return representation

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))


class Abstract_GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Abstract_GCN, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        row, col = edge_index
        if self.aggr == 'mean':
            out = torch.zeros_like(x)
            out.index_add_(0, row, x[col])
            deg = torch.bincount(row, minlength=x.size(0)).unsqueeze(-1).float()
            out /= deg.clamp(min=1)
        else:
            raise NotImplementedError(f"Aggregation method {self.aggr} not implemented")
        return self.linear(out)


class GCN(BasicModel):
    def __init__(self, model_config):
        super(GCN, self).__init__(model_config)
        self.edge_index = model_config['edge_index']
        self.num_user = model_config['num_user']
        self.num_item = model_config['num_item']
        self.dim_output = model_config['dim_output']
        self.dim_input = model_config['dim_input']
        self.dim_latent = model_config['dim_latent']
        self.aggr_mode = model_config['aggr_mode']
        self.concate = model_config['concate']
        self.num_layer = model_config['num_layer']


        self.preference = nn.Embedding(self.num_user, self.dim_latent or self.dim_input).to(self.device)
        nn.init.xavier_normal_(self.preference.weight)
        self.MLP = nn.Linear(self.dim_input, self.dim_latent) if self.dim_latent else None

        self.gcn_layers = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.merge_layers = nn.ModuleList()

        for i in range(self.num_layer):
            dim_in = self.dim_latent if self.dim_latent and i == 0 else self.dim_output
            dim_out = self.dim_output
            self.gcn_layers.append(Abstract_GCN(dim_in, dim_out))
            self.linears.append(nn.Linear(dim_in, dim_out))
            g_layer_in_dim = dim_out + dim_out if self.concate else dim_out
            self.g_layers.append(nn.Linear(g_layer_in_dim, dim_out))

        self._initialize_weights()

    def _initialize_weights(self):
        for gcn_layer in self.gcn_layers:
            nn.init.xavier_normal_(gcn_layer.linear.weight)
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)
        for merge_layer in self.merge_layers:
            nn.init.xavier_normal_(merge_layer.weight)

    def forward(self, features, id_embedding):
        x = self.MLP(features) if self.MLP else features
        x = torch.cat((self.preference.weight, x), dim=0)
        x = F.normalize(x)

        for i in range(self.num_layer):
            h = F.leaky_relu(self.gcn_layers[i](x, self.edge_index))
            x_hat = F.leaky_relu(self.linears[i](x))
            if self.has_id:
                x_hat += id_embedding
            x = F.leaky_relu(self.g_layers[i](torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(
                h + x_hat)

        return x
