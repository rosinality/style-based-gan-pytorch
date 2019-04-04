import torch
from torchvision import utils

from model import StyledGenerator

generator = StyledGenerator(512).cuda()
generator.load_state_dict(torch.load('checkpoint/130000.model'))

mean_style = None

step = 6

shape = 4 * 2 ** step

for i in range(10):
    style = generator.mean_style(torch.randn(1024, 512).cuda())

    if mean_style is None:
        mean_style = style

    else:
        mean_style += style

mean_style /= 10

image = generator(
    torch.randn(50, 512).cuda(),
    step=step,
    alpha=1,
    mean_style=mean_style,
    style_weight=0.7,
)

utils.save_image(image, 'sample.png', nrow=10, normalize=True, range=(-1, 1))

for j in range(20):
    source_code = torch.randn(9, 512).cuda()
    target_code = torch.randn(5, 512).cuda()

    images = [torch.ones(1, 3, shape, shape).cuda() * -1]

    source_image = generator(
        source_code, step=step, alpha=1, mean_style=mean_style, style_weight=0.7
    )
    target_image = generator(
        target_code, step=step, alpha=1, mean_style=mean_style, style_weight=0.7
    )

    images.append(source_image)

    for i in range(5):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(9, 1), source_code],
            step=step,
            alpha=1,
            mean_style=mean_style,
            style_weight=0.7,
            mixing_range=(0, 1),
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    # print([i.shape for i in images])

    images = torch.cat(images, 0)

    utils.save_image(
        images, f'sample_mixing_{j}.png', nrow=10, normalize=True, range=(-1, 1)
    )
