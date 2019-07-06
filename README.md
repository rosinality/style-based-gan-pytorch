# Style-Based GAN in PyTorch

##### Update (2019/07/04)

* Now trainer uses pre-resized lmdb dataset for more stable data loading and training.
* Model architecture is now more closely matches with official implementation.

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

![Sample of the model trained on FFHQ](doc/sample_ffhq_new.png)
![Style mixing sample of the model trained on FFHQ](doc/sample_mixing_ffhq_new.png)

512px sample from the generator trained on FFHQ.
