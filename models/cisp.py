import torch as th
import torch.nn as nn
from .transformers import vit_deit_base_distilled_patch16_224
from .transformers import ShapeTransformer
import numpy as np


class Cisp(nn.Module):

    def __init__(self, embed_dim, im_enc_pretrained=False):
        super().__init__()
        self.image_encoder = vit_deit_base_distilled_patch16_224(pretrained=im_enc_pretrained)
        self.shape_encoder = ShapeTransformer()
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_encoder.embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.shape_proj = nn.Sequential(
            nn.Linear(self.shape_encoder.embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.logit_scale = nn.Parameter(th.ones([]) * np.log(1 / 0.07))

    def forward(self, image, shape):
        image_features, shape_features = self.get_features(image, shape)

        return self.compute_logits(image_features, shape_features)

    def get_features(self, image, shape, cpu=False):
        image_features = self.image_proj(self.image_encoder(image)[:, 0])
        shape_features = self.shape_proj(self.shape_encoder(shape)[:, 0])

        if cpu:
            return image_features.cpu(), shape_features.cpu()
        else:
            return image_features, shape_features

    def compute_logits(self, image_features, shape_features):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        shape_features = shape_features / shape_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ shape_features.t()
        logits_per_shape = logits_per_image.t()

        return logits_per_image, logits_per_shape
