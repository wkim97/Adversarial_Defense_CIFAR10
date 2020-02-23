import os
import pathlib
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import advertorch.attacks as attacks
import matplotlib.pyplot as plt
import numpy as np
from clean_model_training.models import ResNet18
from denoising.DnCNN_model import DnCNN

print(os.getcwd())
os.chdir('/home/sgvr/wkim97/Adversarial_Defense_CIFAR10')

batch_size = 100
image_batch_size = 1
use_gpu = torch.cuda.is_available()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(
    './data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.CIFAR10(
    './data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classification_model_path = './models/CIFAR10_net.pth'

model = ResNet18()
if use_gpu:
    model = model.cuda()
model.load_state_dict(torch.load(classification_model_path))



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
momentum_iterative_attack = attacks.MomentumIterativeAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                                            eps=0.3, nb_iter=40, decay_factor=1.0, eps_iter=0.01,
                                                            clip_min=0.0, clip_max=1.0, targeted=False)

cw_attack = attacks.CarliniWagnerL2Attack(model, num_classes=10, confidence=0, learning_rate=0.01,
                                          binary_search_steps=9, max_iterations=10000, abort_early=True,
                                          initial_const=0.001, clip_min=0.0, clip_max=1.0,
                                          loss_fn=nn.CrossEntropyLoss(reduction="sum"))

l2_pgd_attack = attacks.L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
                                    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
                                    clip_max=1.0, targeted=False)

###################################################################################################
# Set up L1 attacks
###################################################################################################
sparse_attack = attacks.SparseL1DescentAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                              eps=0.3, nb_iter=40, eps_iter=0.01, rand_init=False,
                                              clip_min=0.0, clip_max=1.0, l1_sparsity=0.95)

jsma_attack = attacks.JacobianSaliencyMapAttack(model, num_classes=10, clip_min=0.0, clip_max=1.0,
                                                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                                                theta=1.0, gamma=1.0, comply_cleverhans=False)

elastic_net_attack = attacks.ElasticNetL1Attack(model, num_classes=10, loss_fn=nn.CrossEntropyLoss(reduction="sum"))

ddnl2_attack = attacks.DDNL2Attack(model, nb_iter=100, gamma=0.05, init_norm=1.0, quantize=True,
                                   levels=256, clip_min=0.0, clip_max=1.0, targeted=False,
                                   loss_fn=nn.CrossEntropyLoss(reduction="sum"))

lbfgs_attack = attacks.LBFGSAttack(model, num_classes=10, batch_size=1, binary_search_steps=9,
                                   max_iterations=100, initial_const=0.01, clip_min=0.0, clip_max=1.0,
                                   loss_fn=nn.CrossEntropyLoss(reduction="sum"), targeted=False)

single_pixel_attack = attacks.SinglePixelAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"))

local_search_attack = attacks.LocalSearchAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"))

spatial_transform_attack = attacks.SpatialTransformAttack(model, num_classes=10,
                                                          loss_fn=nn.CrossEntropyLoss(reduction="sum"))


attacks = {"clean" : None,
           "fgsm" : fgsm_attack,
           "bim" : bim_attack,
           "linf_pgd" : linf_pgd_attack,
           "momentum iterative" : momentum_iterative_attack,
           # "cw" : cw_attack,
           "l2_pgd" : l2_pgd_attack,
           "jsma" : jsma_attack,
           "ddnl2" : ddnl2_attack,
           "lbfgs" : lbfgs_attack,
           "single pixel" : single_pixel_attack,
           "spatial transform" : spatial_transform_attack}


def save_image(image, path):
    fig = plt.figure()
    sample = np.transpose(vutils.make_grid(image, normalize=True).cpu().detach().numpy(), (1, 2, 0))
    plt.imsave(path, sample, cmap="gray")
    plt.close(fig)


# def generate_images(attack, targeted):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))])
#
#     testset = torchvision.datasets.MNIST(
#         '../data', train=False, download=True, transform=transform)
#     testloader = torch.utils.data.DataLoader(
#         testset, batch_size=image_batch_size, shuffle=False)
#
#     if targeted:
#         print("Generating images for denoised targeted {} attack".format(attack))
#         store_path = '../data/denoised_MNIST/targeted_{}'.format(attack)
#     else:
#         print("Generating images for denoised untargeted {} attack".format(attack))
#         store_path = '../data/denoised_MNIST/untargeted_{}'.format(attack)
#     attack_func = attacks[attack]
#
#     index = 0
#     with tqdm(total=(1000 - 1000 % image_batch_size)) as _tqdm:
#         for data in testloader:
#             if index == 1000:
#                 break
#             images, labels = data
#             if use_gpu:
#                 images, labels = images.cuda(), labels.cuda()
#             path = store_path + '/{}'.format(labels.detach().cpu().numpy()[0])
#             directory = os.path.dirname(path)
#             if not os.path.exists(directory):
#                 pathlib.Path(path).mkdir(parents=True, exist_ok=True)
#             if attack != "clean":
#                 if targeted:
#                     target = torch.ones_like(labels) * 3
#                     attack_func.targeted = True
#                     noisy_images = attack_func.perturb(images, target)
#                 else:
#                     noisy_images = attack_func.perturb(images, labels)
#             else:
#                 noisy_images = images
#             noisy_images = denoiser(noisy_images)
#             save_image(noisy_images, path + '/{}.png'.format(index))
#             index += 1
#
#             _tqdm.update(image_batch_size)


# attack = function


def test(attack, denoise, targeted, epoch):
    global attacks
    denoising_model_path = './models/dncnn_models/DnCNN_model_{}_epochs.pth'.format(epoch)
    denoiser = DnCNN(num_layers=17, num_features=64)
    if use_gpu:
        denoiser = denoiser.cuda()
    denoiser.load_state_dict(torch.load(denoising_model_path))

    if denoise:
        file_path = './denoising/results/denoised_accuracy_results_{}_epochs.csv'.format(epoch)
    else:
        file_path = './denoising/results/accuracy_results_{}_epochs.csv'.format(epoch)

    ###################################################################################################
    # Create file to store results
    ###################################################################################################
    f = open(file_path, 'a')

    ###################################################################################################
    # Classification results on clean dataset
    ###################################################################################################
    if attack == "clean":
        f.write("Clean dataset\n")
        correct = 0
        total = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with tqdm(total=(len(testset) - len(testset) % batch_size)) as _tqdm:
            if denoise:
                _tqdm.set_description('Epoch {}: Denoised {} attack: '.format(epoch, attack))
            else:
                _tqdm.set_description('Epoch {}: Undenoised {} attack: '.format(epoch, attack))
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
                avg_accuracy = 100 * correct / total
                _tqdm.set_postfix_str('Avg Accuracy: {:2.1f}'.format(avg_accuracy))
                _tqdm.update(batch_size)
        f.write('Avg Accuracy, %.4f %%\n'
                % (100 * float(correct) / total))
        for i in range(10):
            f.write('Accuracy of %s, %.4f %%\n'
                    % (classes[i], 100 * float(class_correct[i]) / class_total[i]))

    ###################################################################################################
    # Classification results on attacked dataset
    ###################################################################################################
    else:
        if targeted:
            f.write("Targeted {} attack\n".format(attack))
        else:
            f.write("Untargeted {} attack\n".format(attack))
        correct = 0
        total = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        attack_func = attacks[attack]

        with tqdm(total=(len(testset) - len(testset) % batch_size)) as _tqdm:
            count = 0
            if targeted and denoise:
                _tqdm.set_description('Epoch {}: Denoised Targeted {} attack: '.format(epoch, attack))
            elif targeted and not denoise:
                _tqdm.set_description('Epoch {}: Undenoised Targeted {} attack: '.format(epoch, attack))
            elif not targeted and denoise:
                _tqdm.set_description('Epoch {}: Denoised Untargeted {} attack: '.format(epoch, attack))
            else:
                _tqdm.set_description('Epoch {}: Undenoised Untargeted {} attack: '.format(epoch, attack))
            for j, data in enumerate(testloader, 0):
                if attack == "cw" and count >= 500:
                    break
                if attack == "jsma" and count >= 2000:
                    break
                images, labels = data
                if use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()
                if targeted:
                    target = torch.ones_like(labels) * 3
                    attack_func.targeted = True
                    noisy_images = attack_func.perturb(images, target)
                else:
                    attack_func.targeted = False
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
                avg_accuracy = 100 * correct / total
                _tqdm.set_postfix_str('Avg Accuracy: {}'.format(avg_accuracy))
                _tqdm.update(batch_size)

                count += batch_size

        f.write('Avg Accuracy, %.4f %%\n'
                % (100 * float(correct) / total))
        for i in range(10):
            f.write('Accuracy of %s, %.4f %%\n'
                    % (classes[i], 100 * float(class_correct[i]) / class_total[i]))

    f.close()


def main():
    # epochs = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    epochs = [500]
    for eps in epochs:
        epoch = eps
        for attack in attacks:
            if attack == "clean":
                test(attack, False, False, epoch)
                test(attack, True, False, epoch)
                # generate_images(attack, False)
            else:
                test(attack, False, False, epoch)
                test(attack, False, True, epoch)
                test(attack, True, False, epoch)
                test(attack, True, True, epoch)
                # generate_images(attack, False)
                # generate_images(attack, True)

main()
