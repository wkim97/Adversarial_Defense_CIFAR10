from __future__ import print_function
import imageio
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from noise_modeling.gan_model import Generator
from noise_modeling.gan_model import Discriminator

selected_class = 0
use_gpu = torch.cuda.is_available()

def generate_noise(selected_class):
    pgd_model_path = './noise_models/pgd_attacked/noise_models/{}'.format(selected_class)
    pgd_generator_path = './gan_models/{}/pgd_generator.pkl'.format(selected_class)
    ngpu = torch.cuda.device_count()

    G = Generator(ngpu)
    G.load_state_dict(torch.load(pgd_generator_path))
    if use_gpu:
        G = G.cuda()

    for j in range(5000):
        path = pgd_model_path + '/noise{}.png'.format(j)
        noise = Variable(torch.randn(1, 100)).cuda()
        image = G(noise)
        fig = plt.figure()
        sample = np.transpose(vutils.make_grid(image, normalize=True).cpu().detach().numpy(), (1, 2, 0))
        plt.imsave(path, sample, cmap="gray")
        plt.close(fig)
        print("Generated noise {} for class {}".format(j, selected_class))


for i in range(10):
    selected_class = i
    generate_noise(selected_class)

