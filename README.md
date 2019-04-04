# Style-Based GAN in PyTorch

Implementation of A Style-Based Generator Architecture for Generative Adversarial Networks (https://arxiv.org/abs/1812.04948) in PyTorch

Usage:

for celebA

> python train.py --mixing -d {folder} PATH

for FFHQ

> python train.py --mixing --loss r1 --sched -d {folder}

## Sample

![Sample of the model trained on CelebA](doc/sample.png)
![Style mixing sample of the model trained on CelebA](doc/sample_mixing.png)

I have mixed styles at 4^2 - 8^2 scale. I can't get samples as dramatic as samles in the original paper. I think my model too dependent on 4^2 scale features - it seems like that much of details determined in that scale, so little variations can be acquired after it.

![Sample of the model trained on FFHQ](doc/sample_ffhq.png)
![Style mixing sample of the model trained on FFHQ](doc/sample_mixing_ffhq.png)

Trained high resolution model on FFHQ. I think result seems more interesting.