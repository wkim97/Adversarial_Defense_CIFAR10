import torch
import torchvision
import torchvision.transforms as transforms
from utils import save_image

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
    train_index = 0
    test_index = 0

    for data in dataloader:
        image, label = data
        if use_gpu:
            image, label = image.cuda(), label.cuda()
        selected_class = label.detach().cpu().numpy()[0]
        read_path = './data/noise_models/noise_models/{}'.format(classes[selected_class])

        if num_per_class[selected_class] < 5000:
            noise_dataset = torchvision.datasets.ImageFolder(
                root=read_path, transform=transforms)
            noise_dataloader = torch.utils.data.DataLoader(
                noise_dataset, batch_size=batch_size, shuffle=True)

            store_path = './data/noisy_CIFAR10/train/generated_noisy_images/images'
            clean_path = './data/noisy_CIFAR10/train/clean_images/images'
            for i in range(10):
                noise, _ = next(iter(noise_dataloader))
                if use_gpu:
                    noise = noise.cuda()
                noisy_image = image + noise
                save_image(noisy_image, store_path + '/{}.png'.format(train_index))
                save_image(image, clean_path + '/{}.png'.format(train_index))
                num_per_class[selected_class] += 1
                train_index += 1
            print("Generating training noisy image {} for class {}".format(train_index,
                                                                           classes[selected_class]))
            print(num_per_class)


        elif test_num_per_class[selected_class] < 1000:
            noise_dataset = torchvision.datasets.ImageFolder(
                root=read_path, transform=transforms)
            noise_dataloader = torch.utils.data.DataLoader(
                noise_dataset, batch_size=batch_size, shuffle=True)

            store_path = './data/noisy_CIFAR10/test/generated_noisy_images/images'
            clean_path = './data/noisy_CIFAR10/test/clean_images/images'
            for i in range(10):
                noise, _ = next(iter(noise_dataloader))
                if use_gpu:
                    noise = noise.cuda()
                noisy_image = image + noise
                save_image(noisy_image, store_path + '/{}.png'.format(test_index))
                save_image(image, clean_path + '/{}.png'.format(test_index))
                test_num_per_class[selected_class] += 1
                test_index += 1
            print("Generating testing noisy image {} for class {}".format(test_index,
                                                                          classes[selected_class]))
            print(test_num_per_class)