import pandas as pd
import torch
import numpy as np
from dataset import BaseDataset
from model import VBPR, MMGCN
from trainer import BPRTrainer


def main():
    model_config = {
        'name': 'VBPR',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dataset': None,
        'embedding_size': 64,
    }

    dataset_config = {
        'name': 'YourDatasetName',
        'device': model_config['device'],
        'split_ratio': 0.8,
        'negative_sample_ratio': 1,
        'shuffle': True,
        'min_inter': 10
    }


    interactions = pd.read_csv('data/elec/elec.inter').values
    dataset = BaseDataset(dataset_config, interactions)
    model_config['dataset'] = dataset


    v_feat = torch.tensor(np.load('data/elec/image_feat.npy'))
    t_feat = torch.tensor(np.load('data/elec/text_feat.npy'))

    if model_config['name'] == 'VBPR':
        model = VBPR(model_config, v_feat=v_feat, t_feat=t_feat)
    elif model_config['name'] == 'MMGCN':
        model = MMGCN(model_config, dataset)

    model.to(model_config['device'])


    trainer = BPRTrainer(model, lr=0.001, batch_size=32, reg_weight=0.01, dataset=dataset)
    trainer.train(epochs=10)


if __name__ == "__main__":
    main()
