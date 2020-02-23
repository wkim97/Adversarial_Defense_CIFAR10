###################################################################################################
# Implementation of Fast Smooth Patch Search Algorithm - noise extraction
# from Image Blind Denoising with GAN Based Noise Modeling - CVPR 2018
# Uses MNIST
###################################################################################################
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import bm3d
import numpy as np
from tqdm import tqdm

pgd_image_path = './data/noisy_images'
pgd_noise_path = './data/noise'
use_gpu = torch.cuda.is_available()
batch_size = 1
mu = 0.1
gamma = 0.25

transforms = transforms.Compose([
     transforms.ToTensor()])
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def extract_pgd(selected_class):
    pgd_dataset = torchvision.datasets.ImageFolder(
        root=pgd_image_path + '/{}'.format(selected_class), transform=transforms)
    pgd_dataloader = torch.utils.data.DataLoader(
        pgd_dataset, batch_size=batch_size, shuffle=False)

    i = 0
    with tqdm(total=(len(pgd_dataset) - len(pgd_dataset) % batch_size)) as _tqdm:
        _tqdm.set_description('Creating noise for {} class: '.format(selected_class))
        for image in pgd_dataloader:
            image = image[0]
            if use_gpu:
                image = image.cuda()

            image = np.transpose(image.squeeze().cpu().numpy(), (1, 2, 0))

            denoised_image = bm3d.bm3d(image + 0.5, sigma_psd=30 / 255,
                                       stage_arg=bm3d.BM3DStages.ALL_STAGES)

            plt.imsave(pgd_noise_path + '/denoised/{}/images/denoised_{}.png'.format(selected_class, i),
                       denoised_image)

            actual_noise = image - denoised_image
            plt.imsave(pgd_noise_path + '/noise/{}/images/noise_{}.png'.format(selected_class, i),
                       actual_noise)
            i += 1
            _tqdm.update(batch_size)

def main():
    for i in range(10):
        selected_class = classes[i]
        extract_pgd(selected_class)