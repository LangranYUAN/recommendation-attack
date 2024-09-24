import numpy as np


class TopKEvaluator:
    def __init__(self, config):
        self.config = config
        self.metrics = config['metrics']
        self.topk = config['topk']
        self._check_args()

    def evaluate(self, batch_matrix_list, eval_data):
        if not batch_matrix_list:  # 检查列表是否为空
            print("Warning: No data to evaluate.")
            return {}  # 返回一个空的字典或其他适当的默认值

        pos_items = eval_data.get_eval_items()
        topk_index = np.concatenate(batch_matrix_list, axis=0)

        assert len(pos_items) == len(topk_index), "用户数量不一致"

        metric_dict = self._calculate_metrics(pos_items, topk_index)
        return metric_dict

    def _check_args(self):
        valid_metrics = {'recall', 'precision', 'ndcg'}
        for m in self.metrics:
            if m.lower() not in valid_metrics:
                raise ValueError(f"Invalid metric '{m}'")
        self.metrics = [metric.lower() for metric in self.metrics]

        if isinstance(self.topk, int):
            self.topk = [self.topk]
        elif isinstance(self.topk, list):
            if any(k <= 0 for k in self.topk):
                raise ValueError('Topk must be a positive integer or list of positive integers.')
        else:
            raise TypeError('Topk must be an integer or list of integers.')

    def _calculate_metrics(self, pos_items, topk_index):
        metric_dict = {}
        for k in self.topk:
            if 'recall' in self.metrics:
                recall = self.recall_at_k(topk_index, pos_items, k)
                metric_dict[f'recall@{k}'] = round(recall, 4)
                print(f"Recall@{k}: {round(recall, 4)}")
            if 'precision' in self.metrics:
                precision = self.precision_at_k(topk_index, pos_items, k)
                metric_dict[f'precision@{k}'] = round(precision, 4)
            if 'ndcg' in self.metrics:
                ndcg = self.ndcg_at_k(topk_index, pos_items, k)
                metric_dict[f'ndcg@{k}'] = round(ndcg, 4)
                print(f"NDCG@{k}: {round(ndcg, 4)}")
        return metric_dict

    def recall_at_k(self, topk_index, pos_items, k):
        recalls = []
        for idx, items in enumerate(topk_index):
            actual_items = pos_items[idx]
            if len(actual_items) == 0:
                continue
            hit = np.isin(items[:k], actual_items)
            recall = np.sum(hit) / len(actual_items)
            recalls.append(recall)
        return np.mean(recalls) if recalls else 0.0

    def precision_at_k(self, topk_index, pos_items, k):
        precisions = []
        for idx, items in enumerate(topk_index):
            actual_items = pos_items[idx]
            if len(actual_items) == 0:
                continue
            hit = np.isin(items[:k], actual_items)
            precision = np.sum(hit) / k
            precisions.append(precision)
        return np.mean(precisions) if precisions else 0.0

    def ndcg_at_k(self, topk_index, pos_items, k):
        ndcgs = []
        for idx, items in enumerate(topk_index):
            actual_items = pos_items[idx]
            if len(actual_items) == 0:
                continue
            dcg = 0.0
            for rank, item in enumerate(items[:k]):
                if item in actual_items:
                    dcg += 1.0 / np.log2(rank + 2)
            idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(actual_items), k))])
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
        return np.mean(ndcgs) if ndcgs else 0.0
