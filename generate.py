import torch
from torchvision import utils

from model import StyledGenerator

device = 'cuda'

generator = StyledGenerator(512).to(device)
generator.load_state_dict(torch.load('checkpoint/180000.model'))
generator.eval()

mean_style = None

step = 7
alpha = 1

shape = 4 * 2 ** step

with torch.no_grad():
    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10

    image = generator(
        torch.randn(15, 512).to(device),
        step=step,
        alpha=alpha,
        mean_style=mean_style,
        style_weight=0.7,
    )

    utils.save_image(image, 'sample.png', nrow=5, normalize=True, range=(-1, 1))

    for j in range(20):
        source_code = torch.randn(5, 512).to(device)
        target_code = torch.randn(3, 512).to(device)

        images = [torch.ones(1, 3, shape, shape).to(device) * -1]

        source_image = generator(
            source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
        )
        target_image = generator(
            target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
        )

        images.append(source_image)

        for i in range(3):
            image = generator(
                [target_code[i].unsqueeze(0).repeat(5, 1), source_code],
                step=step,
                alpha=alpha,
                mean_style=mean_style,
                style_weight=0.7,
                mixing_range=(0, 1),
            )
            images.append(target_image[i].unsqueeze(0))
            images.append(image)

        # print([i.shape for i in images])

        images = torch.cat(images, 0)

        utils.save_image(
            images, f'sample_mixing_{j}.png', nrow=6, normalize=True, range=(-1, 1)
        )
