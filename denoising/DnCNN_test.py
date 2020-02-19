import os
import pathlib
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import advertorch.attacks as attacks
from clean_model_training.models import ResNet18
from denoising.DnCNN_model import DnCNN
from utils import save_image


batch_size = 100
image_batch_size = 1
use_gpu = torch.cuda.is_available()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])
trainset = torchvision.datasets.CIFAR10(
    './data/CIFAR10', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.CIFAR10(
    './data/CIFAR10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True)
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
classification_model_path = './models/CIFAR10_net.pth'
denoising_model_path = './models/DnCNN_model.pth'

model = ResNet18()
if use_gpu:
    model = model.cuda()
model.load_state_dict(torch.load(classification_model_path))

denoiser = DnCNN(num_layers=17, num_features=64)
if use_gpu:
    denoiser = denoiser.cuda()
denoiser.load_state_dict(torch.load(denoising_model_path))

###################################################################################################
# Set up Linf attacks
###################################################################################################
fgsm_attack = attacks.GradientSignAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                                         clip_min=0.0, clip_max=1.0, targeted=False)

bim_attack = attacks.LinfBasicIterativeAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                              eps=0.3, nb_iter=10, eps_iter=0.05, clip_min=0.0,
                                              clip_max=1.0, targeted=False)

linf_pgd_attack = attacks.LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                                        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
                                        clip_max=1.0, targeted=False)

###################################################################################################
# Set up L2 attacks
###################################################################################################
l2_pgd_attack = attacks.L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                                    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
                                    clip_max=1.0, targeted=False)

###################################################################################################
# Set up L1 attacks
###################################################################################################
jsma_attack = attacks.JacobianSaliencyMapAttack(model, num_classes=10, clip_min=0.0, clip_max=1.0,
                                                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                                theta=1.0, gamma=1.0, comply_cleverhans=False)

ddnl2_attack = attacks.DDNL2Attack(model, nb_iter=100, gamma=0.05, init_norm=1.0, quantize=True,
                                   levels=256, clip_min=0.0, clip_max=1.0, targeted=False,
                                   loss_fn=nn.CrossEntropyLoss(reduction="sum"))

lbfgs_attack = attacks.LBFGSAttack(model, num_classes=10, batch_size=1, binary_search_steps=9,
                                   max_iterations=100, initial_const=0.01, clip_min=0.0, clip_max=1.0,
                                   loss_fn=nn.CrossEntropyLoss(reduction="sum"), targeted=False)


attacks = {"clean" : None,
           "fgsm" : fgsm_attack,
           "bim" : bim_attack,
           "linf_pgd" : linf_pgd_attack,
           "l2_pgd" : l2_pgd_attack,
           "jsma" : jsma_attack,
           "ddnl2" : ddnl2_attack,
           "lbfgs" : lbfgs_attack}


def generate_images(attack, targeted):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    testset = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=image_batch_size, shuffle=False)

    if targeted:
        print("Generating images for denoised targeted {} attack".format(attack))
        store_path = './data/denoised_CIFAR10/targeted_{}'.format(attack)
    else:
        print("Generating images for denoised untargeted {} attack".format(attack))
        store_path = './data/denoised_CIFAR10/untargeted_{}'.format(attack)
    attack_func = attacks[attack]

    index = 0
    with tqdm(total=(1000 - 1000 % image_batch_size)) as _tqdm:
        for data in testloader:
            if index == 1000:
                break
            images, labels = data
            if use_gpu:
                images, labels = images.cuda(), labels.cuda()
            path = store_path + '/{}'.format(labels.detach().cpu().numpy()[0])
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            if attack != "clean":
                if targeted:
                    target = torch.ones_like(labels) * 3
                    attack_func.targeted = True
                    noisy_images = attack_func.perturb(images, target)
                else:
                    noisy_images = attack_func.perturb(images, labels)
            else:
                noisy_images = images
            noisy_images = denoiser(noisy_images)
            save_image(noisy_images, path + '/{}.png'.format(index))
            index += 1

            _tqdm.update(image_batch_size)


# attack = function
def test(attack, denoise, targeted):
    global attacks
    if denoise:
        file_path = './denoising/results/denoised_accuracy_results.csv'
    else:
        file_path = './denoising/results/accuracy_results.csv'

    ###################################################################################################
    # Create file to store results
    ###################################################################################################
    f = open(file_path, 'a')

    ###################################################################################################
    # Classification results on clean dataset
    ###################################################################################################
    if attack == "clean":
        f.write("Clean dataset\n")
        if denoise:
            print("Denoised clean dataset...")
        else:
            print("Clean dataset...")
        correct = 0
        total = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        for j, data in enumerate(testloader, 0):
            images, labels = data
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()
            if denoise:
                images = denoiser(images)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        f.write('Avg Accuracy, %d %%\n'
                % (100 * correct / total))
        print('Avg Accuracy, %d %%\n'
                % (100 * correct / total))
        for i in range(10):
            f.write('Accuracy of %s, %2d %%\n'
                    % (classes[i], 100 * class_correct[i] / class_total[i]))

    ###################################################################################################
    # Classification results on attacked dataset
    ###################################################################################################
    else:
        if targeted:
            f.write("Targeted {} attack\n".format(attack))
        else:
            f.write("Untargeted {} attack\n".format(attack))
        if denoise and targeted:
            print("Denoised targeted {} attack...".format(attack))
        elif denoise and not targeted:
            print("Denoised untargeted {} attack...".format(attack))
        elif not denoise and targeted:
            print("Targeted {} attack...".format(attack))
        else:
            print("Untargeted {} attack...".format(attack))
        correct = 0
        total = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        attack_func = attacks[attack]

        for j, data in enumerate(testloader, 0):
            images, labels = data
            if use_gpu:
                images = images.cuda()
                labels = labels.cuda()
            if targeted:
                target = torch.ones_like(labels) * 3
                attack_func.targeted = True
                noisy_images = attack_func.perturb(images, target)
            else:
                noisy_images = attack_func.perturb(images, labels)
            if denoise:
                noisy_images = denoiser(noisy_images)
            outputs = model(noisy_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        f.write('Avg Accuracy, %d %%\n'
                % (100 * correct / total))
        print('Avg Accuracy, %d %%\n'
                % (100 * correct / total))
        for i in range(10):
            f.write('Accuracy of %s, %2d %%\n'
                    % (classes[i], 100 * class_correct[i] / class_total[i]))

    f.close()


def main():
    for attack in attacks:
        if attack == "clean":
            test(attack, False, False)
            test(attack, True, False)
            # generate_images(attack, False)
        else:
            test(attack, False, False)
            test(attack, False, True)
            test(attack, True, False)
            test(attack, True, True)
            # generate_images(attack, False)
            # generate_images(attack, True)
