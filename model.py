import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class BasicModel(nn.Module):
    def __init__(self, model_config):
        super(BasicModel, self).__init__()
        self.model_fig = model_config
        self.name = model_config['name']
        self.device = model_config['device']
        self.dataset = model_config['dataset']
        self.n_users = self.dataset.n_users
        self.n_items = self.dataset.n_items

        self.v_feat, self.t_feat = None, None

        self.init_parameters()

    def init_parameters(self):
        raise NotImplementedError("Subclasses should initialize model parameters.")

    def get_rep(self):
        raise NotImplementedError

    def bpr_forward(self, users,  pos_items, neg_items):
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

    def predict(self,users):
        rep = self.get_rep()
        user_rep = rep[users, :]
        all_items_rep = rep[self.n_users:, :]
        scores = torch.mm(user_rep, all_items_rep.t())
        return scores


class VBPR(BasicModel):
    def __init__(self, model_config, v_feat=None, t_feat=None):
        super(VBPR, self).__init__(model_config)
        self.v_feat = v_feat
        self.t_feat = t_feat

        # load parameters info
        self.embedding_size = model_config['embedding_size']

        # Define layers
        if self.v_feat is not None and self.t_feat is not None:
            self.item_raw_features = torch.cat((self.t_feat, self.v_feat), -1)
        elif self.v_feat is not None:
            self.item_raw_features = self.v_feat
        elif self.t_feat is not None:
            self.item_raw_features = self.t_feat
        else:
            self.item_raw_features = None  # 当没有特征时的处理

        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.embedding_size)
        self.u_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.embedding_size * 2)))
        self.i_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.embedding_size)))

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_normal_(self.u_embedding)
        nn.init.xavier_normal_(self.i_embedding)
        nn.init.xavier_normal_(self.item_linear.weight)
        if self.item_linear.bias is not None:
            nn.init.zeros_(self.item_linear.bias)

    def get_rep(self, dropout=0.0):
        user_rep = F.dropout(self.u_embedding, dropout)  # 用户嵌入
        item_feat_rep = self.item_linear(self.item_raw_features)  # 物品动态特征表示
        item_emb_rep = F.dropout(self.i_embedding, dropout)  # 物品静态嵌入表示

        # 合并用户表示和物品表示
        item_rep = torch.cat((item_emb_rep, item_feat_rep), dim=1)
        return user_rep, item_rep

    def forward(self, dropout=0.0):
        user_rep, item_rep = self.get_rep(dropout)
        return user_rep, item_rep

class MMGCN(BasicModel):
    def __init__(self, model_config, dataset):
        super(MMGCN, self).__init__(model_config)
        dim_x = model_config['embedding_size']
        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)

        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0], :]), dim=1)
        self.num_modal = 0


        if self.v_feat is not None:
            self.v_gcn = GCN(self.edge_index, self.num_user, self.v_feat.size(1), dim_x,
                             self.aggr_mode, self.concate, self.num_layer)
            self.num_modal += 1

        if self.t_feat is not None:
            self.t_gcn = GCN(self.edge_index, self.num_user, self.num_item, self.t_feat.size(1), dim_x,
                             self.aggr_mode, self.concate, self.num_layer)
            self.num_modal += 1

        self.id_embedding = nn.init.xavier_normal_(
            torch.rand((self.num_user + self.num_item, dim_x), requires_grad=True)).to(self.device)

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
    def __init__(self, edge_index, dim_input, dim_output, concate, num_layer,
                 has_id, dim_latent):
        super(GCN, self).__init__()
        self.dim_output = dim_output
        self.dim_input = dim_input
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id

        self.preference = nn.Embedding(self.num_user, self.dim_latent or self.dim_input).to(self.device)
        nn.init.xavier_normal_(self.preference.weight)
        self.MLP = nn.Linear(self.dim_input, self.dim_latent) if self.dim_latent else None

        self.gcn_layers = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.merge_layers = nn.ModuleList()

        for i in range(num_layer):
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
