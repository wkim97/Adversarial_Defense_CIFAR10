import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from clean_model_training.models import ResNet18
from utils import AverageMeter

batch_size = 100
use_gpu = torch.cuda.is_available()
num_epochs = 100

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(
    '../data/CIFAR10', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.CIFAR10(
    '../data/CIFAR10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True)
model_path = '../models/CIFAR10_net.pth'
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


model = ResNet18()
if use_gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

###################################################################################################
# Train the model
###################################################################################################
def train():
    for epoch in range(num_epochs):
        epoch_losses = AverageMeter()
        with tqdm(total=(len(trainset) - len(trainset) % batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, num_epochs))
            for i, data in enumerate(trainloader, 0):
                images, labels = data
                if use_gpu:
                    images = images.cuda()
                    labels = labels.cuda()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_losses.update(loss.item(), len(images))
                loss.backward()
                optimizer.step()
                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(images))

    torch.save(model.state_dict(), model_path)

def test():
    file_path = './results/accuracy_on_clean_model.csv'
    f = open(file_path, 'w')
    f.write("Clean dataset\n")

    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for j, data in enumerate(testloader, 0):
        images, labels = data
        if use_gpu:
            images = images.cuda()
            labels = labels.cuda()
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

train()
test()
