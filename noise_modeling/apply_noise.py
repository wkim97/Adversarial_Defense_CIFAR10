###################################################################################################
# Applies noise to MNIST dataset
###################################################################################################
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import advertorch.attacks as attacks
from clean_model_training.models import ResNet18
from utils import save_image

batch_size = 1
use_gpu = torch.cuda.is_available()
pgd_path = './data/noisy_images'
model_path = './models/CIFAR10_net.pth'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(
    './data/CIFAR10', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


model = ResNet18()
if use_gpu:
    model = model.cuda()
model.load_state_dict(torch.load(model_path))
linf_pgd_attack = attacks.LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                                        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
                                        clip_max=1.0, targeted=False)

def apply_pgd(selected_class):
    i = 0
    for data in trainloader:
        images, labels = data
        if labels.numpy()[0] == selected_class:
            path = pgd_path + '/{}/images/{}.png'.format(classes[selected_class], i)
            if use_gpu:
                images, labels = images.cuda(), labels.cuda()
            images = linf_pgd_attack.perturb(images, labels)
            save_image(images, path)
            i += batch_size
            print("PGD image {} for class {} created".format(i + 1, classes[selected_class]))
            if i == 5000:
                break

def main():
    for i in range(10):
        selected_class = i
        apply_pgd(selected_class)