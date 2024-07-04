import argparse
from matplotlib import gridspec, pyplot as plt
import numpy as np
import os
import random
import torch as th
import torchvision.transforms as transforms
from tqdm import trange

from models.diffusion_utils import GaussianDiffusion, get_named_beta_schedule, p_xt
from models import ic3d
from data.datasets import ShapeNetDataset
import data.transforms as tf

np.random.seed(42)
th.manual_seed(42)
random.seed(42)
N_STEPS = 1000
W = 1.5

image_trans = transforms.Compose([
    tf.to_numpy,
    tf.RandomBackground(((240, 240), (240, 240), (240, 240))),
    tf.CenterCrop((224, 224), (128, 128)),
    tf.ToTensor(),
    lambda x: x[0],
    tf.normalize])


def main(dataroot, taxonomies, cisp_emb_path_test, batch_size, model_path, sample_idx, samples_to_generate, save_path,
         cuda):
    # set device
    device = "cuda" if cuda == 'True' else "cpu"

    # load dataset and model
    dataset_paths = [os.path.join(dataroot, 'ShapeNet.json'), os.path.join(dataroot, 'ShapeNetVoxel/ShapeNetVox32'),
                     os.path.join(dataroot, 'ShapeNetImage/ShapeNetRendering')]
    dataset = ShapeNetDataset(*dataset_paths, image_trans, split='test', mode='first', taxonomies=taxonomies,
                              cisp_embeddings_path=cisp_emb_path_test)

    model = ic3d(attention_resolutions=[16, 8, 4], num_classes=len(taxonomies) + 1).to(device)
    model.load_state_dict(th.load(model_path))

    # setup diffusion
    beta_schedule = get_named_beta_schedule("linear", num_diffusion_timesteps=N_STEPS)
    n = GaussianDiffusion(betas=beta_schedule)

    im_num = -1
    generated_images = [-1]
    f = 0
    while f < samples_to_generate:
        # sample random images or use idx given as argument
        curr_batch = []
        while len(curr_batch) < batch_size:
            if sample_idx is None:
                while im_num in generated_images:
                    im_num = np.random.randint(0, len(dataset))
            else:
                im_num = sample_idx
            generated_images.append(im_num)
            curr_batch.append(im_num)

        gt = []
        image = []

        for id in curr_batch:
            gt.append(dataset[id]['voxel'])
            image.append(dataset[id]['image'])

        # get images and gt shapes
        gt = th.stack(gt)
        image = th.stack(image).to(device)

        # sample random noise x0
        x = th.randn(batch_size, 1, 32, 32, 32)  # Start with random noise
        n_steps = n.num_timesteps

        x = x.to(device)
        voxs = []

        # get cisp embeddings for guidance
        emb_unnoised_cisp = dataset.cisp_embeddings[curr_batch].to(device)

        # perform n steps of the backward diffusion process
        for i in trange(n_steps):
            t = th.tensor(n_steps - i - 1, dtype=th.long).to(device)
            with th.no_grad():
                uncond_token = model.null_token
                uncond_token = uncond_token.repeat(batch_size, 1)
                pred_noise_cond = model(x, t.unsqueeze(0), image, emb_unnoised_cisp,
                                        th.tensor([0]).unsqueeze(0).to(device))
                pred_noise_uncond = model(x, t.unsqueeze(0), image, uncond_token,
                                          th.tensor([1]).unsqueeze(0).to(device))
                pred_noise = (1 + W) * pred_noise_cond - W * pred_noise_uncond
                x = p_xt(x, beta_schedule, n, pred_noise, t.unsqueeze(0))

        voxs.append(x.cpu())

        # save result images and shapes
        for id in range(batch_size):
            # plot gt shape and query image
            gt_id = gt[id].squeeze().cpu()
            gs = gridspec.GridSpec(1, len(voxs) + 2)
            fig = plt.figure(figsize=(12, 5))
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_aspect('auto')
            ax1.imshow(th.permute(dataset[curr_batch[id]]['image'].squeeze(), (1, 2, 0)))
            ax1 = fig.add_subplot(gs[0, 1], projection='3d')
            ax1.set_aspect('auto')
            ax1.voxels(th.permute(gt_id, (0, 2, 1)), facecolors=[1, 0, 0, 0.8], edgecolors=[0, 1, 0, 0])
            x = voxs[0][id].squeeze()

            # binarize data
            x[x > 0.5] = 1
            x[x != 1] = 0

            # plot generated shape
            ax1 = fig.add_subplot(gs[0, 2], projection='3d')
            ax1.set_aspect('auto')
            ax1.voxels(th.permute(x, (0, 2, 1)), facecolors=[1, 0, 0, 0.8], edgecolors=[0, 1, 0, 0])

            # save the plot and the shape
            plt.savefig(os.path.join(save_path, str(f) + '.png'),
                        bbox_inches='tight')
            th.save(x.to_sparse(),
                    os.path.join(save_path, str(f)))
            plt.close(fig)
            f += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='dataset', help='root to Shapenet directory.')
    parser.add_argument('--taxonomies', nargs='+', default=['aeroplane', 'car', 'chair', 'table', 'watercraft'],
                        help='taxonomies to consider')
    parser.add_argument('--cisp_emb_path_test', required=True, help='root to pre-trained cisp embeddings (test '
                                                                    'split).')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size.')
    parser.add_argument('--model_path', type=str, required=True, help='path to ic3d saved state dictionary.')
    parser.add_argument('--sample_idx', type=int, default=None, help='ShapeNet sample number to condition generation.')
    parser.add_argument('--samples_to_generate', type=int, required=True, help='number of samples to generate.')
    parser.add_argument('--save_path', type=str, default='', help='directory to store models and figures.')
    parser.add_argument('--cuda', default='True', choices=['True', 'False'], help='whether to use gpu or not.')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
