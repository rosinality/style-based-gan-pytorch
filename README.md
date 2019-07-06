# Style-Based GAN in PyTorch

Implementation of A Style-Based Generator Architecture for Generative Adversarial Networks (https://arxiv.org/abs/1812.04948) in PyTorch

Usage:

for celebA

> python train.py --mixing -d {folder} PATH

for FFHQ

> python train.py --mixing --loss r1 --sched -d {folder}

## Sample

![Sample of the model trained on FFHQ](doc/sample_ffhq_new.png)
![Style mixing sample of the model trained on FFHQ](doc/sample_mixing_ffhq_new.png)

512px sample from the generator trained on FFHQ.