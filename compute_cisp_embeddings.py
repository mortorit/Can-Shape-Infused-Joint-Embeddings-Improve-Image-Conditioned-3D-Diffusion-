import argparse
import os
import torch as th
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data import transforms as tf
from data.datasets import ShapeNetDataset
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

def get_device(cuda):
    return "cuda" if cuda == 'True' else "cpu"

def load_model(model_path, embed_dim, device):
    model = Cisp(embed_dim=embed_dim).to(device)
    model.load_state_dict(th.load(model_path))
    model.eval()
    return model

def get_dataloader(dataroot, taxonomies, image_trans, batch_size, split):
    dataset_paths = [
        os.path.join(dataroot, 'ShapeNet.json'),
        os.path.join(dataroot, 'ShapeNetVoxel/ShapeNetVox32'),
        os.path.join(dataroot, 'ShapeNetImage/ShapeNetRendering')
    ]
    dataset = ShapeNetDataset(*dataset_paths, image_trans, split=split, mode='first', taxonomies=taxonomies)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, len(dataset)

def compute_and_save_embeddings(model, dataloader, device, save_path, split, batch_size, dataset_length):
    image_features = th.empty((dataset_length, 256)).to(device)
    shape_features = th.empty((dataset_length, 256)).to(device)

    with th.no_grad():
        for i, data in enumerate(dataloader):
            print(f"{i}/{len(dataloader)}")
            im, shape = data['image'].to(device), data['voxel'].unsqueeze(1).to(device)
            image_features[i * batch_size:i * batch_size + batch_size], shape_features[i * batch_size:i * batch_size + batch_size] = model.get_features(im, shape, cpu=True)

    th.save(image_features, os.path.join(save_path, f"{split}_im_emb"))
    th.save(shape_features, os.path.join(save_path, f"{split}_shape_emb"))

def main(dataroot, taxonomies, model_path, embed_dim, batch_size, split, save_path, cuda):
    device = get_device(cuda)
    image_trans = get_image_transforms()
    model = load_model(model_path, embed_dim, device)
    dataloader, dataset_length = get_dataloader(dataroot, taxonomies, image_trans, batch_size, split)
    compute_and_save_embeddings(model, dataloader, device, save_path, split, batch_size, dataset_length)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='dataset')
    parser.add_argument('--taxonomies', nargs='+', default=['aeroplane', 'car', 'chair', 'table', 'watercraft'], help='taxonomies to consider')
    parser.add_argument('--model_path', type=str, required=True, help='path to pretrained CISP model')
    parser.add_argument('--embed_dim', type=int, default=256, help='size of the embeddings.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to process data with.')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'], help='the dataset split.')
    parser.add_argument('--save_path', type=str, default='', help='directory to store models and figures.')
    parser.add_argument('--cuda', default='True', choices=['True', 'False'], help='whether to use gpu or not.')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
