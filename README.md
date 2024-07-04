# Can Shape-Infused Joint Embeddings Improve Image-Conditioned 3D Diffusion?

<p align="center">
  <img src="images/pipeline_image.svg" alt="Sublime's custom image"/>
</p>

This directory contains the code for our [paper](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=yhvx_3AAAAAJ&citation_for_view=yhvx_3AAAAAJ:zYLM7Y9cAGgC)
 presented at IJCNN 2024 @ Yokohama, Japan.


The file `requirements.txt` contains the pip requirements.
The folder `models` contains the PyTorch implementations of the proposed models, CISP and IC3D.
The folder `data` contains python scripts for handling ShapeNet data.
The folder `dataset` has to contain the ShapeNet dataset, organized as follows:

      /dataset
          |-ShapeNetImage (renderings: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz)
          |-ShapeNetVoxel (voxelized models: http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz)

You can pretrain the CISP model by running the [train_cisp.py](train_cisp.py) script.

You can then run the scripts to compute [CISP embeddings](compute_cisp_embeddings.py) and [CLIP embeddings](compute_clip_embeddings.py).

Once embeddings for the dataset have been generated, you can train the diffusion model by running its [training script](train_ic3d.py)

Each script takes command line arguments to control settings, most of which are defaulted to the values in the paper, however make sure to check them for a correct use.

For our validation, we used the metrics 1-NNA, MMD and COV implemented in: https://github.com/alexzhou907/PVD
 
Pre-trained weights will be publicly released.
