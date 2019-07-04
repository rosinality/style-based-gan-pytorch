# Style-Based GAN in PyTorch

Implementation of A Style-Based Generator Architecture for Generative Adversarial Networks (https://arxiv.org/abs/1812.04948) in PyTorch

Usage:

You should prepare lmdb dataset

> python prepare_data.py --out LMDB_PATH --n_worker N_WORKER DATASET_PATH

This will convert images to jpeg and pre-resizes it. (For example, 8/16/32/64/128/256/512/1024) Then you can train StyleGAN.

for celebA

> python train.py --mixing LMDB_PATH

for FFHQ

> python train.py --mixing --loss r1 --sched LMDB_PATH

## Sample

![Sample of the model trained on CelebA](doc/sample.png)
![Style mixing sample of the model trained on CelebA](doc/sample_mixing.png)

I have mixed styles at 4^2 - 8^2 scale. I can't get samples as dramatic as samles in the original paper. I think my model too dependent on 4^2 scale features - it seems like that much of details determined in that scale, so little variations can be acquired after it.

![Sample of the model trained on FFHQ](doc/sample_ffhq.png)
![Style mixing sample of the model trained on FFHQ](doc/sample_mixing_ffhq.png)

Trained high resolution model on FFHQ. I think result seems more interesting.
