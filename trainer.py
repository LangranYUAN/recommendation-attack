import torch
import torch.nn.functional as F


class BPRTrainer:
    def __init__(self, trainer_config):
        self.model = trainer_config['model']
        self.optimizer = trainer_config['optimizer']
        self.dataloader = trainer_config['dataloader']
        self.reg_weight = trainer_config['reg_weight']
        self.device = next(self.model.parameters()).device
        self.evaluator = trainer_config['evaluator']  # 使用TopKEvaluator
        self.topk = trainer_config['topk']  # Top-K 参数

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

            total_loss += loss.item()

        return total_loss

    def evaluate(self, eval_dataloader):
        self.model.eval()
        batch_matrix_list = []
        all_pos_items = []
        pos_len_list = []

        for batch_users, batch_pos_items, _ in eval_dataloader:
            batch_users = batch_users.to(self.device)
            batch_pos_items = batch_pos_items.to(self.device)

            all_pos_items.extend([[item.item()] for item in batch_pos_items])
            pos_len_list.extend([1] * len(batch_pos_items))

            # 模型对所有物品进行评分预测
            scores = self.model.predict(batch_users)

            # 处理空的 scores
            if scores.size(0) == 0 or len(batch_pos_items) == 0:
                continue

            if batch_pos_items.max() >= scores.size(1):
                continue
                
            scores[torch.arange(scores.size(0)), batch_pos_items] = float('-inf')

            topk_indices = torch.topk(scores, max(self.topk), dim=1)
            batch_matrix_list.append(topk_indices.cpu().numpy())

        eval_data = EvalData(all_pos_items, pos_len_list)

        results = self.evaluator.evaluate(batch_matrix_list, eval_data)
        return results

    def train(self, epochs, eval_dataloader=None, eval_steps=1):
        for epoch in range(epochs):
            avg_loss = self.train_epoch()
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

            # 每隔 eval_steps 评估一次
            if eval_dataloader and (epoch + 1) % eval_steps == 0:
                eval_results = self.evaluate(eval_dataloader)
                print(f"Epoch {epoch + 1}, Evaluation Results: {eval_results}")


class EvalData:
    def __init__(self, pos_items, pos_len_list):
        self.pos_items = pos_items
        self.pos_len_list = pos_len_list

    def get_eval_items(self):
        return self.pos_items

    def get_eval_len_list(self):
        return self.pos_len_list

