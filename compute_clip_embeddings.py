import argparse
import os
import torch as th
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data import transforms as tf
from data.datasets import ShapeNetDataset
import open_clip

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

def load_model(device):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to(device)
    model.eval()
    return model, preprocess

def get_dataloader(dataroot, taxonomies, preprocess, batch_size, split):
    dataset_paths = [os.path.join(dataroot, 'ShapeNet.json'), os.path.join(dataroot, 'ShapeNetVox32'), os.path.join(dataroot, 'ShapeNetRendering')]
    dataset = ShapeNetDataset(*dataset_paths, preprocess, split=split, mode='first', taxonomies=taxonomies)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, len(dataset)

def compute_and_save_embeddings(model, dataloader, device, save_path, split, batch_size, dataset_length):
    image_features = th.empty((dataset_length, 512)).to(device)
    with th.no_grad(), th.cuda.amp.autocast():
        for i, data in enumerate(dataloader):
            print(f"{i}/{len(dataloader)}")
            im = data['image'].to(device).float()
            image_features[i * batch_size:i * batch_size + batch_size] = model.encode_image(im)
    th.save(image_features, os.path.join(save_path, f"{split}_im_emb_clip"))

def main(dataroot, taxonomies, batch_size, split, save_path, cuda):
    device = get_device(cuda)
    model, preprocess = load_model(device)
    dataloader, dataset_length = get_dataloader(dataroot, taxonomies, preprocess, batch_size, split)
    compute_and_save_embeddings(model, dataloader, device, save_path, split, batch_size, dataset_length)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='dataset')
    parser.add_argument('--taxonomies', nargs='+', default=['aeroplane', 'car', 'chair', 'table', 'watercraft'], help='taxonomies to consider')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to process data with.')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'], help='the dataset split.')
    parser.add_argument('--save_path', type=str, default='', help='directory to store models and figures.')
    parser.add_argument('--cuda', default='True', choices=['True', 'False'], help='whether to use gpu or not.')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
