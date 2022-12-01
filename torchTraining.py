import torch
from models.ImagenetSubset import ImagenetSubset
from torch import nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from vit_pytorch import ViT

import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        self.flatten = nn.Flatten()
        self.net_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
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
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def show_data(train_loader):
    cifar100_classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                        'bowl',
                        'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
                        'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur',
                        'dolphin',
                        'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
                        'lamp',
                        'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle',
                        'mountain',
                        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                        'pickup_truck',
                        'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray',
                        'road',
                        'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                        'spider',
                        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
                        'television',
                        'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree',
                        'wolf',
                        'woman', 'worm']
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    classes = cifar10_classes

    dataiter = iter(train_loader)
    images, labels = dataiter.__next__()
    images = images.numpy()

    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20 // 2, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx].squeeze().T, cmap='gray')
        ax.set_title(classes[labels[idx]])

    plt.show()


def main():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(224, 4, pad_if_needed=True),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    print('loading training datasets...')
    train_data = ImagenetSubset(
        "datasets/imagenet/Data/train",
        transform=transform_train,
        subset_file="./datasets/imagenet/SubSets/imagenet_50")

    val_data = ImagenetSubset(
        "datasets/imagenet/Data/val",
        transform=transform_test,
        subset_file="./datasets/imagenet/SubSets/imagenet_50")
    '''
    train_data = datasets.ImageNet(
        root="datasets",
        train=True,
        download=True,
        transform=transform_train
    )

    print('loading test datasets...')
    test_data = datasets.ImageNet(
        root="datasets",
        train=False,
        download=True,
        transform=transform_test
    )
    '''

    batch_size = 64
    epochs = 120

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # show_data(train_loader)

    for inputs, label in train_loader:
        print(f'inputs shape: {inputs.shape}')
        print(f'label shape: {label.shape}')
        break

    print('initializing model...')
    print(f'using device: {device}')
    # my_resent18 = models.resnet18(pretrained=False, num_classes=10).to(device)
    # my_net = MyNet().to(device)
    # resnet18 = models.resnet18(num_classes=50)
    vitNet = ViT(
        image_size=224,
        patch_size=4,
        num_classes=50,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1
    )
    model = vitNet.cuda()
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    print('training...')
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_net(model, criterion, optimizer, train_loader)
        test_net(model, val_loader, criterion)
        scheduler.step()


if __name__ == '__main__':
    main()
