###################################################################################################
# Code to generate noise models from noise images produced from noise_extraction.py
# Uses GAN to generate noise models
###################################################################################################
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
from utils import save_image

###################################################################################################
# Initialize file paths and other variables
###################################################################################################
selected_class = 0
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def train(selected_class):
    pgd_model_path = './data/noise_models/noise_models/{}/images'.format(selected_class)
    pgd_training_path = './data/noise_models/training_steps/{}'.format(selected_class)
    pgd_dataset_path = './data/noise/noise/{}'.format(selected_class)
    pgd_generator_path = './models/gan_models/{}/pgd_generator.pkl'.format(selected_class)
    pgd_discriminator_path = './models/gan_models/{}/pgd_discriminator.pkl'.format(selected_class)
    use_gpu = torch.cuda.is_available()

    num_epochs = 500
    batch_size = 25
    ngpu = torch.cuda.device_count()

    #############################################################################################################
    # Call in data
    #############################################################################################################
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    pgd_dataset = torchvision.datasets.ImageFolder(
        root=pgd_dataset_path, transform=transform)
    pgd_dataloader = torch.utils.data.DataLoader(
        pgd_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


    #############################################################################################################
    # Implementation - weights initialization
    # weights_init takes a model as an input and re-initializes all layers
    #############################################################################################################
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    G = Generator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 2):
        G = nn.DataParallel(G, list(range(ngpu)))
    G.apply(weights_init)
    print(G)

    D = Discriminator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 2):
        D = nn.DataParallel(D, list(range(ngpu)))
    D.apply(weights_init)
    print(D)

    if use_gpu:
        G = G.cuda()
        D = D.cuda()


    #############################################################################################################
    # Criterion and optimizer
    #############################################################################################################
    criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))


    #############################################################################################################
    # Visualizing results
    #############################################################################################################
    def square_plot(data, path):
        if type(data) == list:
            data = np.concatentate(data)
        data = (data - data.min()) / (data.max() - data.min())
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = (((0, n ** 2 - data.shape[0]),
                    (0, 1), (0, 1)) + ((0, 0),) * (data.ndim - 3))
        data = np.pad(data, padding, mode='constant', constant_values=1)
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                               + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        fig = plt.imsave(path, data)
        plt.close(fig)


    #############################################################################################################
    # Training
    #############################################################################################################
    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    generated_images = []

    z_fixed = Variable(torch.randn(5 * 5, 100))
    if use_gpu:
        z_fixed = z_fixed.cuda()

    img_list = []
    iters = 0

    for epoch in range(num_epochs):
        D_losses = []
        G_losses = []
        dataloader = pgd_dataloader
        model_path = pgd_model_path
        training_path = pgd_training_path
        generator_path = pgd_generator_path
        discriminator_path = pgd_discriminator_path

        for i, (real_data, _) in enumerate(dataloader):
            real_data = Variable(real_data)
            label_real = Variable(torch.ones(batch_size))
            label_fake = Variable(torch.zeros(batch_size))
            if use_gpu:
                real_data, label_real, label_fake = real_data.cuda(), label_real.cuda(), label_fake.cuda()
            z = Variable(torch.randn((batch_size, 100)))
            if use_gpu:
                z = z.cuda()
            fake_data = G(z)

            D.zero_grad()
            real_output = D(real_data)
            D_loss_real = criterion(real_output, label_real)

            fake_output = D(fake_data)
            D_loss_fake = criterion(fake_output, label_fake)
            D_loss = D_loss_real + D_loss_fake
            D_loss.backward()
            D_optimizer.step()
            D_x = real_output.mean().item()

            D_losses.append(D_loss.data)

            z = Variable(torch.randn((batch_size, 100)))
            if use_gpu:
                z = z.cuda()
            fake_data = G(z)
            fake_output = D(fake_data)
            G.zero_grad()
            G_loss = criterion(fake_output, label_real)

            G_loss.backward()
            G_optimizer.step()
            D_G_z = fake_output.mean().item()
            G_losses.append(G_loss.data)

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)):%.4f'
                      % (epoch + 1, num_epochs, i, len(dataloader),
                         D_loss.item(), G_loss.item(), D_x, D_G_z))
                G_losses.append(G_loss.item())
                D_losses.append(D_loss.item())
                if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                    with torch.no_grad():
                        fake = G(z_fixed).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                iters += 1

        # Save individual images at the end of training
        if epoch == num_epochs - 1:
            for j in range(5000):
                path = model_path + '/noise{}.png'.format(j)
                noise = Variable(torch.randn(1, 100)).cuda()
                image = G(noise)
                save_image(image, path)

        true_positive_rate = (real_output > 0.5).float().mean().data  # Probability real image classified as real
        true_negative_rate = (fake_output < 0.5).float().mean().data  # Probability fake image classified as fake
        base_message = ("Epoch: {epoch:<3d} D_Loss: {d_loss:<8.6} G_Loss: {g_loss:<8.6} "
                        "True Positive Rate: {tpr:<5.1%} True Negative Rate: {tnr:<5.1%}")
        message = base_message.format(
            epoch=epoch,
            d_loss=sum(D_losses) / len(D_losses),
            g_loss=sum(G_losses) / len(G_losses),
            tpr=true_positive_rate,
            tnr=true_negative_rate)
        print(message)

        fake_data_fixed = G(z_fixed)
        image_path = training_path + '/epoch{}.png'.format(epoch)
        square_plot(fake_data_fixed.view(25, 28, 28).cpu().data.numpy(), path=image_path)
        generated_images.append(image_path)

        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

    torch.save(G.state_dict(), generator_path)
    torch.save(D.state_dict(), discriminator_path)

    generated_image_array = [imageio.imread(generated_image) for generated_image in generated_images]
    imageio.mimsave(training_path + '/GAN_generation.gif', generated_image_array, fps=5)

def main():
    for i in range(10):
        selected_class = classes[i]
        train(selected_class)