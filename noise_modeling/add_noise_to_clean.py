import os
import torch
import torchvision
import torchvision.transforms as transforms
from utils import save_image
from tqdm import tqdm

os.chdir('/home/sgvr/wkim97/Adversarial_Defense_CIFAR10')

use_gpu = torch.cuda.is_available()
batch_size = 1

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = torchvision.datasets.CIFAR10(
    './data/CIFAR10', train=True, download=True, transform=transforms)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True)
num_per_class = [0] * 10
test_num_per_class = [0] * 10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def main():
    train_index = 1
    class_num = 500
    noise_num = 40

    with tqdm(total=(class_num * noise_num * 10 - class_num * noise_num * 10 % train_index)) as _tqdm:
        _tqdm.set_description('Applying noise model')
        for data in dataloader:
            image, label = data
            if use_gpu:
                image, label = image.cuda(), label.cuda()
            selected_class = label.detach().cpu().numpy()[0]
            read_path = './data/noise_models/noise_models/{}'.format(classes[selected_class])

            if num_per_class[selected_class] < class_num:
                noise_dataset = torchvision.datasets.ImageFolder(
                    root=read_path, transform=transforms)
                noise_dataloader = torch.utils.data.DataLoader(
                    noise_dataset, batch_size=batch_size, shuffle=True)

                store_path = './data/noisy_CIFAR10/train/generated_noisy_images/images'
                clean_path = './data/noisy_CIFAR10/train/clean_images/images'
                for i in range(noise_num):
                    noise, _ = next(iter(noise_dataloader))
                    if use_gpu:
                        noise = noise.cuda()
                    noisy_image = image + noise
                    save_image(noisy_image, store_path + '/{}.png'.format(train_index))
                    save_image(image, clean_path + '/{}.png'.format(train_index))
                    train_index += 1
                num_per_class[selected_class] += 1
                _tqdm.update(batch_size * 100)


        # elif test_num_per_class[selected_class] < 1000:
        #     noise_dataset = torchvision.datasets.ImageFolder(
        #         root=read_path, transform=transforms)
        #     noise_dataloader = torch.utils.data.DataLoader(
        #         noise_dataset, batch_size=batch_size, shuffle=True)
        #
        #     store_path = './data/noisy_CIFAR10/test/generated_noisy_images/images'
        #     clean_path = './data/noisy_CIFAR10/test/clean_images/images'
        #     for i in range(10):
        #         noise, _ = next(iter(noise_dataloader))
        #         if use_gpu:
        #             noise = noise.cuda()
        #         noisy_image = image + noise
        #         save_image(noisy_image, store_path + '/{}.png'.format(test_index))
        #         save_image(image, clean_path + '/{}.png'.format(test_index))
        #         test_num_per_class[selected_class] += 1
        #         test_index += 1
        #     print("Generating testing noisy image {} for class {}".format(test_index,
        #                                                                   classes[selected_class]))
        #     print(test_num_per_class)


main()
