from models.ResNet import resnet18
import torch
from torch import nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        self.flatten = nn.Flatten()
        self.net_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.net_stack(x)
        return x


def train_net(model, criterion, optimizer, train_loader):
    size = len(train_loader.dataset)
    model.train()

    for batch, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_net(model, test_loader, criterion):
    size = len(test_loader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += criterion(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def show_data(train_loader):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy()

    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx].squeeze().T, cmap='gray')
        ax.set_title(classes[labels[idx]])

    plt.show()


def main():

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    print('downloading training data...')
    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform_train
    )

    print('downloading test data...')
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform_test
    )

    batch_size = 64

    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    show_data(train_loader)

    for inputs, label in train_loader:
        print(f'inputs shape: {inputs.shape}')
        print(f'label shape: {label.shape}')
        break

    print('initializing model...')
    print(f'using device: {device}')
    # resent18 = models.resnet18(pretrained=False, num_classes=10).to(device)
    # my_net = MyNet().to(device)
    # my_resnet =

    model = resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 70], gamma=.1)

    print('training...')
    epochs = 90
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_net(model, criterion, optimizer, train_loader)
        test_net(model, test_loader, criterion)

    # train_net(model, criterion, optimizer, train_loader)
    #
    # print('testing...')
    # test_net(model, test_loader, criterion)


if __name__ == '__main__':
    main()
