import torch
from dataset import AmazonDataset
from model import VBPR, MMGCN
from trainer import BPRTrainer
import torch.optim as optim
from torch.utils.data import DataLoader
from topk_evaluator import TopKEvaluator


def main():
    dataset_config = {
        'name': 'elec',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'split_ratio': 0.8,
        'negative_sample_ratio': 1,
        'min_inter': 5,
        'image_feat_path': 'data/elec/image_feat.npy',
        'text_feat_path': 'data/elec/text_feat.npy',
        'interactions_path': 'data/elec/elec.inter'
    }
    dataset = AmazonDataset(dataset_config)

    train_dataset, eval_dataset = dataset.split(dataset_config['split_ratio'])

    model_config = {
        'name': 'VBPR',
        'device':  dataset_config['device'],
        'dataset': dataset,
        'embedding_size': 64,
        'v_feat': dataset.v_feat,
        't_feat': dataset.t_feat,
        'dim_input': 128,
        'dim_output': 64,
        'concate': True,
        'has_id': False,
        'num_layer': 3,
        'dim_latent': 256,
        'aggr_mode': 'mean'
    }

    if model_config['name'] == 'VBPR':
        model = VBPR(model_config)
    elif model_config['name'] == 'MMGCN':
        model = MMGCN(model_config)

    model.to(model_config['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    config = {
        'metrics': ['recall', 'precision', 'ndcg'],
        'topk': [10, 20]
    }
    evaluator = TopKEvaluator(config=config)

    trainer_config = {
        'model': model,
        'optimizer': optimizer,
        'dataloader': dataloader,
        'reg_weight': 0.01,
        'evaluator': evaluator,
        'topk': config['topk']
    }

    trainer = BPRTrainer(trainer_config)
    trainer.train(epochs=10, eval_dataloader=eval_dataloader, eval_steps=1)

if __name__ == "__main__":
    main()
