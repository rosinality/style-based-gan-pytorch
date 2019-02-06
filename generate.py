import torch
from torchvision import utils

generator = torch.load('checkpoint/600000.model')

mean_style = generator.mean_style(torch.randn(4096, 512).cuda())
image = generator(torch.randn(50, 512).cuda(), step=5, alpha=1, mean_style=mean_style, style_weight=0.7)

utils.save_image(image, f'sample.png', nrow=10, normalize=True, range=(-1, 1))