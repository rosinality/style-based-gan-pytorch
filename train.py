from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import StyledGenerator, Discriminator


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def lsun_loader(path):
    def loader(transform):
        data = datasets.LSUNClass(
            path, transform=transform,
            target_transform=lambda x: 0)
        data_loader = DataLoader(data, shuffle=False, batch_size=batch_size,
                                 num_workers=4)

        return data_loader

    return loader


def celeba_loader(path):
    def loader(transform):
        data = datasets.ImageFolder(path, transform=transform)
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size,
                                 num_workers=4)

        return data_loader

    return loader


def sample_data(dataloader, image_size=4):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    loader = dataloader(transform)

    return loader

    # for img, label in loader:
    #     yield img, label


def train(generator, discriminator, loader, options):
    step = options.init_size // 4 - 1
    data_loader = sample_data(loader, 4 * 2 ** step)
    dataset = iter(data_loader)
    pbar = tqdm(range(600000))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    one = torch.FloatTensor([1]).cuda()
    mone = one * -1
    iteration = 0

    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, 0.00002 * iteration)

        if iteration > 100000:
            alpha = 0
            iteration = 0
            step += 1

            if step > 5:
                alpha = 1
                step = 5
            dataset = sample_data(loader, 4 * 2 ** step)

        try:
            real_image, label = next(dataset)

        except (OSError, StopIteration):
            dataset = iter(data_loader)
            real_image, label = next(dataset)

        iteration += 1

        b_size = real_image.size(0)
        real_image = real_image.cuda()
        label = label.cuda()
        real_predict = discriminator(
            real_image, step=step, alpha=alpha)
        real_predict = real_predict.mean() \
            - 0.001 * (real_predict ** 2).mean()
        real_predict.backward(mone)

        fake_image = generator(
            torch.randn(b_size, code_size).cuda(),
            step=step, alpha=alpha)
        fake_predict = discriminator(
            fake_image, step=step, alpha=alpha)
        fake_predict = fake_predict.mean()
        fake_predict.backward(one)

        eps = torch.rand(b_size, 1, 1, 1).cuda()
        x_hat = eps * real_image.data + (1 - eps) * fake_image.data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat, step=step, alpha=alpha)
        grad_x_hat = grad(
            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1)
                         .norm(2, dim=1) - 1)**2).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        grad_loss_val = grad_penalty.item()
        disc_loss_val = (real_predict - fake_predict).item()

        d_optimizer.step()

        if (i + 1) % n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(
                torch.randn(batch_size, code_size).cuda(),
                step=step, alpha=alpha)

            predict = discriminator(fake_image, step=step, alpha=alpha)

            loss = -predict.mean()
            gen_loss_val = loss.item()

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i + 1) % 100 == 0:
            images = []
            for _ in range(5):
                images.append(g_running(
                    torch.randn(5 * 10, code_size).cuda(),
                    step=step, alpha=alpha).data.cpu())
            utils.save_image(
                torch.cat(images, 0),
                f'sample/{str(i + 1).zfill(6)}.png',
                nrow=5 * 10,
                normalize=True,
                range=(-1, 1))

        if (i + 1) % 10000 == 0:
            torch.save(g_running, f'checkpoint/{str(i + 1).zfill(6)}.model')

        pbar.set_description(
            (f'{i + 1}; G: {gen_loss_val:.5f}; D: {disc_loss_val:.5f};'
             f' Grad: {grad_loss_val:.5f}; Alpha: {alpha:.3f}'))



if __name__ == '__main__':
    code_size = 512
    batch_size = 16
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--init-size', default=8, type=int, help='initial image size')
    parser.add_argument('-d', '--data', default='celeba', type=str,
                        choices=['celeba', 'lsun'],
                        help=('Specify dataset. '
                            'Currently CelebA and LSUN is supported'))

    args = parser.parse_args()

    generator = StyledGenerator(code_size).cuda()
    discriminator = Discriminator().cuda()
    g_running = StyledGenerator(code_size).cuda()
    g_running.train(False)

    class_loss = nn.CrossEntropyLoss()

    g_optimizer = optim.Adam(generator.generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    g_optimizer.add_param_group({'params': generator.style.parameters(), 'lr': args.lr * 0.01})
    d_optimizer = optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))


    accumulate(g_running, generator, 0)


    if args.data == 'celeba':
        loader = celeba_loader(args.path)

    elif args.data == 'lsun':
        loader = lsun_loader(args.path)

    train(generator, discriminator, loader, args)
