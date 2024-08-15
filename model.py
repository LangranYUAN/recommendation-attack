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
        self.batch_size = model_config['train_batch_size']
        self.num_users = model_config['num_users']
        self.num_items = model_config['num_items']
        self.n_users = self.dataset.n_users
        self.n_items = self.dataset.n_items

        self.v_feat, self.t_feat = None, None

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
            l2_norm_sq += torch.norm(self.v_gcn.preference, p=2) ** 2

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
        self.u_embedding_size = self.i_embedding_size = model_config['embedding_size']

        # Initialize embeddings
        self.u_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(model_config['num_users'], self.u_embedding_size * 2)))
        self.i_embedding = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(model_config['num_items'], self.i_embedding_size)))

        # Feature handling
        self.v_feat = v_feat
        self.t_feat = t_feat
        self.item_raw_features = self._combine_features()

        # Define layers
        self.item_linear = nn.Linear(self.item_raw_features.shape[1], self.i_embedding_size)

    def _combine_features(self):
        if self.v_feat is not None and self.t_feat is not None:
            return torch.cat((self.t_feat, self.v_feat), -1)
        elif self.v_feat is not None:
            return self.v_feat
        elif self.t_feat is not None:
            return self.t_feat
        else:
            return None

    def get_rep(self,  dropout=0.0):
        user_rep = F.dropout(self.u_embedding, dropout)
        item_rep = self.item_linear(self.item_raw_features)
        return torch.cat((user_rep, item_rep), dim=1)



class MMGCN(BasicModel):
    def __init__(self, model_config, dataset):
        super(MMGCN, self).__init__(model_config)
        dim_x = model_config['embedding_size']
        num_layer = model_config['n_layers']
        has_id = True
        self.weight = torch.tensor([[1.0], [-1.0]]).to(self.device)

        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0], :]), dim=1)
        self.num_modal = 0


        if self.v_feat is not None:
            self.v_gcn = GCN(self.edge_index, self.batch_size, self.num_user, self.num_item, self.v_feat.size(1), dim_x,
                             self.aggr_mode,
                             self.concate, num_layer=num_layer, has_id=has_id, device=self.device)
            self.num_modal += 1

        if self.t_feat is not None:
            self.t_gcn = GCN(self.edge_index, self.batch_size, self.num_user, self.num_item, self.t_feat.size(1), dim_x,
                             self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, device=self.device)
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

class GCN(BasicModel):
    def __init__(self, model_config, edge_index, dim_feat, dim_id, aggr_mode, concate, num_layer,
                 has_id, dim_latent=None):
        super(GCN, self).__init__(model_config)
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id


        self.preference = nn.init.xavier_normal_(torch.rand((self.num_user, self.dim_latent or self.dim_feat), requires_grad=True)).to(self.device)
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
