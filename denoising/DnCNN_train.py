import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from denoising.DnCNN_model import DnCNN
from utils import AverageMeter

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
ngpu = torch.cuda.device_count()

batch_size = 1
lr = 0.001
batch_size = 50
num_epochs = 150

def show_image(image):
    plt.figure()
    sample = np.transpose(vutils.make_grid(image, normalize=True).cpu().detach().numpy(), (1, 2, 0))
    plt.imshow(sample)
    plt.show()

def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    noisy_dataset = torchvision.datasets.ImageFolder(
        root='./data/noisy_CIFAR10/train/generated_noisy_images', transform=transform)
    noisy_dataloader = torch.utils.data.DataLoader(
        noisy_dataset, batch_size=batch_size, shuffle=False)
    clean_dataset = torchvision.datasets.ImageFolder(
        root='./data/noisy_CIFAR10/train/clean_images', transform=transform)
    clean_dataloader = torch.utils.data.DataLoader(
        clean_dataset, batch_size=batch_size, shuffle=False)

    model = DnCNN(num_layers=17)
    if use_gpu:
        model = model.to(device)
    if (device.type == 'cuda') and (ngpu > 2):
        model = nn.DataParallel(model, list(range(ngpu)))
    criterion = nn.MSELoss(size_average=False, reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_losses = AverageMeter()

        with tqdm(total=(len(noisy_dataset) - len(noisy_dataset) % batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, num_epochs))
            for i, data in enumerate(zip(noisy_dataloader, clean_dataloader)):
                noisy_image = data[0][0]
                clean_image = data[1][0]
                if use_gpu:
                    noisy_image, clean_image = noisy_image.to(device), clean_image.to(device)

                preds = model(noisy_image)

                loss = criterion(preds, clean_image) / (2 * len(noisy_image))

                epoch_losses.update(loss.item(), len(noisy_image))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(noisy_image))


    torch.save(model.state_dict(), os.path.join('./models', 'DnCNN_model.pth'))


def main():
    train()