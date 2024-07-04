import json
import os
from typing import Callable, List, Optional, Union
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from .binvox_rw import read_as_3d_array

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ShuffleDataset(Dataset):
    def __init__(self, dataset: Dataset):
        super(ShuffleDataset, self).__init__()
        self._dataset = dataset
        self.indices = torch.randperm(len(dataset))

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict:
        return self._dataset[self.indices[index].item()]

class ShapeNetDataset(Dataset):
    def __init__(
        self,
        annot_path: str,
        model_path: str,
        image_path: str,
        image_transforms: Callable,
        split: str = 'train',
        mode: str = 'random',
        background: tuple = (255, 255, 255),
        view_num: int = 1,
        taxonomies: Optional[List[str]] = None,
        im_embs_path: Optional[str] = None
    ):
        """
        Initializes the ShapeNetDataset.

        @param annot_path: Path to the "ShapeNet.json" file.
        @param model_path: Path to the "ShapeNetVox32" folder.
        @param image_path: Path to the "ShapeNetRendering" folder.
        @param image_transforms: Preprocessing transformations for images.
        @param split: One of "train", "val", "test".
        @param mode: 'random' to load a random image if there are multiple, 'first' to always load the first images.
        @param view_num: Number of views to load at once. -1 for single view.
        @param taxonomies: List of taxonomies to filter.
        @param im_embs_path: Path to precomputed image embeddings.
        """
        self._validate_init_params(split, mode)
        self._meta_data, self.taxonomies_names, self.num_classes = self._load_annotations(
            annot_path, split, taxonomies
        )
        self.im_embs = torch.load(im_embs_path) if im_embs_path else None
        self._model_path = model_path
        self._image_path = image_path
        self._image_transforms = image_transforms
        self._mode = mode
        self._background = background
        self._view_num = view_num

    def _validate_init_params(self, split: str, mode: str) -> None:
        if split not in ['train', 'val', 'test']:
            raise ValueError(f'Unexpected split: {split}')
        if mode not in ['random', 'first']:
            raise ValueError(f'Unexpected mode: {mode}')

    def _load_annotations(
        self, annot_path: str, split: str, taxonomies: Optional[List[str]]
    ) -> Union[List[dict], List[str], int]:
        with open(annot_path) as annot_file:
            annots = json.load(annot_file)
        meta_data = []
        taxonomies_names = []
        for taxonomy in annots:
            if taxonomies and taxonomy['taxonomy_name'] not in taxonomies:
                continue
            for model_id in taxonomy[split]:
                meta_data.append({
                    'taxonomy_id': taxonomy['taxonomy_id'],
                    'taxonomy_name': taxonomy['taxonomy_name'],
                    'model_id': model_id
                })
            taxonomies_names.append(taxonomy['taxonomy_name'])
        return meta_data, taxonomies_names, len(annots)

    def _load_voxel_data(self, binvox_path: str) -> np.ndarray:
        with open(binvox_path, 'rb') as f:
            raw_voxel = read_as_3d_array(f)
        return raw_voxel.data.astype(np.float32), raw_voxel

    def _get_image_indices(self, image_file_list: List[str]) -> torch.Tensor:
        if self._mode == 'random':
            return torch.randperm(len(image_file_list))[:self._view_num]
        return torch.arange(len(image_file_list))[:self._view_num]

    def _load_images(self, image_base_path: str, image_file_list: List[str], image_indices: torch.Tensor) -> torch.Tensor:
        images = [
            self._image_transforms(Image.open(os.path.join(image_base_path, image_file_list[i.item()])))
            for i in image_indices
        ]
        images = torch.stack(images)
        return images.squeeze(0) if self._view_num == 1 else images

    def __getitem__(self, index: int) -> dict:
        meta_data = self._meta_data[index]
        taxonomy_id = meta_data['taxonomy_id']
        model_id = meta_data['model_id']
        binvox_path = os.path.join(self._model_path, taxonomy_id, model_id, 'model.binvox')
        voxel, raw_voxel = self._load_voxel_data(binvox_path)
        image_base_path = os.path.join(self._image_path, taxonomy_id, model_id, 'rendering')
        image_file_list = sorted(f for f in os.listdir(image_base_path) if f not in ['rendering_metadata.txt', 'renderings.txt'])
        image_indices = self._get_image_indices(image_file_list)
        images = self._load_images(image_base_path, image_file_list, image_indices)
        return {
            'image': images,
            'id': index,
            'model_id': model_id,
            'taxonomy_id': taxonomy_id,
            'taxonomy_name': meta_data['taxonomy_name'],
            'scale': raw_voxel.scale,
            'translate': torch.Tensor(raw_voxel.translate),
            'voxel': torch.Tensor(voxel).unsqueeze(0)[0, :, :, :],
            'taxonomy_class_id': self.taxonomies_names.index(meta_data['taxonomy_name']),
            'im_emb_': self.im_embs[index] if self.im_embs is not None else 0
        }

    def __len__(self) -> int:
        return len(self._meta_data)
