import torch
from dataset import AmazonDataset
from model import VBPR, MMGCN
from trainer import BPRTrainer
import torch.optim as optim
from torch.utils.data import DataLoader


def main():
    dataset_config = {
        'name': 'Elec',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'split_ratio': 0.8,
        'negative_sample_ratio': 1,
        'min_inter': 5,
        'image_feat_path': 'data/elec/image_feat.npy',
        'text_feat_path': 'data/elec/text_feat.npy',
        'interactions_path': 'data/elec/elec.inter'
    }
    dataset = AmazonDataset(dataset_config)

    model_config = {
        'name': 'VBPR',
        'device':  dataset_config['device'],
        'dataset': 'elec',
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
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    trainer_config = {
        'model': model,
        'optimizer': optimizer,
        'dataloader': dataloader,
        'reg_weight': 0.01
    }

    trainer = BPRTrainer(trainer_config)
    trainer.train(epochs=10)

if __name__ == "__main__":
    main()
