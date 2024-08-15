import torch
import torch.nn.functional as F

class BPRTrainer:
    def __init__(self, model, optimizer, dataloader, reg_weight=0.01):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.reg_weight = reg_weight
        self.device = next(model.parameters()).device

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for users, pos_items, neg_items in self.dataloader:
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            self.optimizer.zero_grad()

            users_r, pos_items_r, neg_items_r, l2_norm_sq = \
                self.model.bpr_forward(users, pos_items, neg_items)

            pos_scores = torch.sum(users_r * pos_items_r, dim=1)
            neg_scores = torch.sum(users_r * neg_items_r, dim=1)
            bpr_loss = F.softplus(neg_scores - pos_scores).mean()

            reg_loss = self.reg_weight * l2_norm_sq.mean()

            loss = bpr_loss + reg_loss

            loss.backward()
            self.optimizer.step()

            # 累加损失
            total_loss += loss.item()

        return total_loss / len(self.dataloader)

    def train(self, epochs):
        for epoch in range(epochs):
            avg_loss = self.train_epoch()
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")


