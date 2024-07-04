import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from time import sleep
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from data.datasets import ShapeNetDataset
import data.transforms as tf
from models import Cisp

def get_image_transforms():
    return transforms.Compose([
        tf.to_numpy,
        tf.RandomBackground(((240, 240), (240, 240), (240, 240))),
        tf.CenterCrop((224, 224), (128, 128)),
        tf.ToTensor(),
        lambda x: x[0],
        tf.normalize
    ])

def setup_device(cuda: str) -> str:
    return "cuda" if cuda == 'True' else "cpu"

def load_datasets(dataroot: str, taxonomies: list, image_trans: transforms.Compose) -> tuple:
    dataset_paths = [
        os.path.join(dataroot, 'ShapeNet.json'),
        os.path.join(dataroot, 'ShapeNetVox32'),
        os.path.join(dataroot, 'ShapeNetRendering')
    ]
    train = ShapeNetDataset(*dataset_paths, image_trans, taxonomies=taxonomies)
    val = ShapeNetDataset(*dataset_paths, image_trans, split='val', mode='first', taxonomies=taxonomies)
    return train, val

def create_dataloaders(train: Dataset, val: Dataset, batch_size: int) -> tuple:
    train_loader = DataLoader(train, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val, shuffle=True, batch_size=batch_size)
    return train_loader, val_loader

def setup_model(embed_dim: int, device: str) -> th.nn.Module:
    model = Cisp(embed_dim=embed_dim).to(device)
    return model

def setup_optimizer_and_loss(model: th.nn.Module) -> tuple:
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-6)
    criterion = th.nn.CrossEntropyLoss()
    return optimizer, criterion

def train_epoch(model: th.nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: th.nn.Module, device: str) -> float:
    model.train()
    epoch_losses = []
    with tqdm(train_loader, unit="batch") as tepoch:
        for data in tepoch:
            im, shape = data['image'].to(device), data['voxel'].unsqueeze(1).to(device)
            logits_i, logits_s = model(im, shape)
            labels = th.arange(im.shape[0]).to(device)
            loss = (criterion(logits_i, labels) + criterion(logits_s, labels)) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if model.logit_scale > np.log(100):
                model.logit_scale = th.nn.Parameter(th.log(th.Tensor([100])))
            epoch_losses.append(loss.item())
            tepoch.set_postfix(training_loss=loss.item())
    return np.mean(epoch_losses)

def validate_epoch(model: th.nn.Module, val_loader: DataLoader, criterion: th.nn.Module, device: str) -> float:
    model.eval()
    epoch_losses = []
    with tqdm(val_loader, unit="batch") as tepoch:
        for data in tepoch:
            im, shape = data['image'].to(device), data['voxel'].unsqueeze(1).to(device)
            logits_i, logits_s = model(im, shape)
            labels = th.arange(im.shape[0]).to(device)
            loss = (criterion(logits_i, labels) + criterion(logits_s, labels)) / 2
            epoch_losses.append(loss.item())
            tepoch.set_postfix(validation_loss=loss.item())
    return np.mean(epoch_losses)

def plot_losses(train_losses: list, val_losses: list, save_path: str) -> None:
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'losses.png'), bbox_inches='tight')

def main(dataroot: str, taxonomies: list, batch_size: int, epochs: int, embed_dim: int, save_path: str, cuda: str) -> None:
    device = setup_device(cuda)
    image_trans = get_image_transforms()
    train, val = load_datasets(dataroot, taxonomies, image_trans)
    train_loader, val_loader = create_dataloaders(train, val, batch_size)
    model = setup_model(embed_dim, device)
    optimizer, criterion = setup_optimizer_and_loss(model)

    train_losses = []
    val_losses = []
    best_loss = float('inf')

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        print(f"Training loss: {train_loss}")

        val_loss = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Validation loss: {val_loss}")

        if val_loss < best_loss:
            best_loss = val_loss
            th.save(model.state_dict(), os.path.join(save_path, f'ep_{epoch}'))

    plot_losses(train_losses, val_losses, save_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='../ShapeNet', help='root to Shapenet directory.')
    parser.add_argument('--taxonomies', nargs='+', default=['aeroplane', 'car', 'chair', 'table', 'watercraft'], help='taxonomies to consider')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size.')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for.')
    parser.add_argument('--embed_dim', type=int, default=256, help='size of the embeddings.')
    parser.add_argument('--save_path', type=str, default='', help='directory to store models and figures.')
    parser.add_argument('--cuda', default='True', choices=['True', 'False'], help='whether to use gpu or not.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
