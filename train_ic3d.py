import argparse
import os
import torch as th
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.datasets import ShapeNetDataset
import data.transforms as tf
from models import ic3d
from models.diffusion_utils import get_named_beta_schedule, GaussianDiffusion
import wandb

def init_wandb(wandb_key):
    wandb.login(key=wandb_key)
    wandb.init(project="ic3d", entity="ic3d")

def get_device(cuda):
    return "cuda" if cuda == 'True' else "cpu"

def get_image_transforms():
    return transforms.Compose([
        tf.to_numpy,
        tf.RandomBackground(((240, 240), (240, 240), (240, 240))),
        tf.CenterCrop((224, 224), (128, 128)),
        tf.ToTensor(),
        lambda x: x[0],
        tf.normalize
    ])

def get_dataloaders(dataroot, taxonomies, cisp_emb_path_train, cisp_emb_path_val, batch_size, image_trans):
    dataset_paths = [os.path.join(dataroot, 'ShapeNet.json'), os.path.join(dataroot, 'ShapeNetVox32'), os.path.join(dataroot, 'ShapeNetRendering')]
    train_data = ShapeNetDataset(*dataset_paths, image_trans, taxonomies=taxonomies, im_embs_path=cisp_emb_path_train)
    val_data = ShapeNetDataset(*dataset_paths, image_trans, split='val', taxonomies=taxonomies, im_embs_path=cisp_emb_path_val)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, train_data

def get_model(train_data, taxonomies, device):
    model = ic3d(attention_resolutions=[16, 8, 4], num_classes=len(taxonomies) + 1, im_emb_dim=train_data.im_embs.shape[1]).to(device)
    return model

def train_epoch(model, train_loader, criterion, opt, scaler, noise, device, N_STEPS, UNCOND_PROB):
    model.train()
    train_loss = 0
    for i, data in enumerate(tqdm(train_loader, unit="batch", desc="Training")):
        x, images, clip_embs = data['voxel'].to(device), data['image'].to(device), data['cisp_embedding'].to(device)
        t = th.randint(0, N_STEPS, (x.shape[0],), dtype=th.long).to(device)
        xt, n = noise.q_sample(x, t).unsqueeze(1)
        bern = th.distributions.Bernoulli(1 - UNCOND_PROB).sample((clip_embs.shape[0], 1)).to(device)
        clip_embs = bern * clip_embs + (1 - bern) * model.null_token
        bern_im = th.distributions.Bernoulli(1 - UNCOND_PROB).sample((clip_embs.shape[0], 1, 1)).to(device)
        with th.autocast(device_type=device, dtype=th.float16, enabled=True):
            out = model(xt, t, images, clip_embs, bern_im).squeeze()
            loss = criterion(out, n)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        train_loss += loss.item()
        tqdm.write(f"Batch {i+1}, Training Loss: {loss.item()}")
    return train_loss / (i + 1)

def validate_epoch(model, val_loader, criterion, noise, device, N_STEPS, UNCOND_PROB):
    model.eval()
    val_loss = 0
    with th.no_grad():
        for i, data in enumerate(tqdm(val_loader, unit="batch", desc="Validation")):
            x, images, clip_embs = data['voxel'].to(device), data['image'].to(device), data['cisp_embedding'].to(device)
            t = th.randint(0, N_STEPS, (x.shape[0],), dtype=th.long).to(device)
            xt, n = noise.q_sample(x, t).unsqueeze(1)
            bern = th.distributions.Bernoulli(1 - UNCOND_PROB).sample((clip_embs.shape[0], 1)).to(device)
            clip_embs = bern * clip_embs + (1 - bern) * model.null_token
            bern_im = th.distributions.Bernoulli(1 - UNCOND_PROB).sample((clip_embs.shape[0], 1, 1)).to(device)
            out = model(xt, t, images, clip_embs, bern_im).squeeze()
            loss = criterion(out, n)
            val_loss += loss.item()
            tqdm.write(f"Batch {i+1}, Validation Loss: {loss.item()}")
    return val_loss / (i + 1)

def save_model(model, save_path, epoch, best_loss, val_loss):
    if val_loss < best_loss:
        th.save(model.state_dict(), os.path.join(save_path, f'ep_{epoch}'))
        return val_loss
    return best_loss

def main(dataroot, taxonomies, cisp_emb_path_train, cisp_emb_path_val, batch_size, epochs, lr, save_path, cuda, wandb_key):
    init_wandb(wandb_key)
    device = get_device(cuda)
    image_trans = get_image_transforms()
    train_loader, val_loader, train_data = get_dataloaders(dataroot, taxonomies, cisp_emb_path_train, cisp_emb_path_val, batch_size, image_trans)
    model = get_model(train_data, taxonomies, device)
    criterion = th.nn.MSELoss()
    opt = th.optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(opt, 'min', patience=7)
    scaler = th.cuda.amp.GradScaler(enabled=True)
    best_loss = float('inf')
    beta_schedule = get_named_beta_schedule("linear", num_diffusion_timesteps=1000)
    noise = GaussianDiffusion(betas=beta_schedule)
    N_STEPS, UNCOND_PROB = 1000, 0.1

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, opt, scaler, noise, device, N_STEPS, UNCOND_PROB)
        val_loss = validate_epoch(model, val_loader, criterion, noise, device, N_STEPS, UNCOND_PROB)
        best_loss = save_model(model, save_path, epoch, best_loss, val_loss)
        wandb.log({"Training Loss": train_loss, "Validation Loss": val_loss})
        scheduler.step(val_loss)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='dataset', help='root to Shapenet directory.')
    parser.add_argument('--taxonomies', nargs='+', default=['aeroplane', 'car', 'chair', 'table', 'watercraft'], help='taxonomies to consider')
    parser.add_argument('--im_emb_path_train', required=True, help='root to pre-trained cisp/clip embeddings (training split).')
    parser.add_argument('--im_emb_path_val', required=True, help='root to pre-trained cisp/clip embeddings (validation split).')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size.')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for.')
    parser.add_argument('--lr', type=int, default=3e-4, help='learning rate.')
    parser.add_argument('--save_path', type=str, default='', help='directory to store models and figures.')
    parser.add_argument('--cuda', default='True', choices=['True', 'False'], help='whether to use gpu or not.')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
