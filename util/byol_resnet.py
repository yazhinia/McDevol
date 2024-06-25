#!/usr/bin/env python
""" run byol training """

import os
import argparse
import random
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, DistributedSampler
import logging

from torchvision import models, transforms, datasets
from torchvision import transforms as T


def split_dataset(dataset, flag_test):
    """ split dataset into training and validation. Also, test set (if flag_test is true)"""
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    if flag_test:
        val_size = np.ceil((total_size - train_size) / 2).astype(int)
        test_size = np.floor((total_size - train_size) / 2).astype(int)
        # return random_split(dataset, [train_size, val_size, test_size])
        return random_split(dataset, [train_size, total_size-train_size, 0])
    else:
        val_size = np.ceil(total_size - train_size).astype(int)
        return random_split(dataset, [train_size, val_size])

def MLP(dim, projection_size, hidden_size=4096):
    " return multiple linear perceptron layer "

    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.LeakyReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        """ update target network by moving average of online network """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def target_update_moving_average(ema_updater, online_encode, \
    online_project, target_encode, target_project):
    """ update moving average """

    for online_params1, online_params2, target_params1, target_params2 \
        in zip(online_encode.parameters(), online_project.parameters(), \
        target_encode.parameters(), target_project.parameters()):
        target_encode.data = ema_updater.update_average(target_params1.data, online_params1.data)
        target_project.data = ema_updater.update_average(target_params2.data, online_params2.data)

class BYOLmodel(nn.Module):
    """ train BYOL model """
    def __init__(
        self,
        args,
        seed: int = 0,
    ):

        torch.manual_seed(seed)

        super(BYOLmodel, self).__init__()

        # Initialize simple attributes
        self.num_workers= 6 if args.cuda else 12
        self.outdir = args.outdir
        self.logger = args.logger
        self.usecuda = args.cuda
        projection_size = 256 # 512 #
        projection_hidden_size = 4096
        self.image_size = 256
        # weights = ResNet50_Weights.DEFAULT
        # preprocess = weights.transforms(antialias=True)
        resnet18 = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.online_encoder = resnet18
        self.indim = self.online_encoder.fc.in_features
        print(self.indim, 'self indimension')
        self.online_encoder.fc = nn.Identity()
        total_params = sum(p.numel() for p in resnet18.parameters() if p.requires_grad)
        print(f"ResNet-18 Total Parameters: {total_params}")



        # Dataset and DataLoader
        transform = transforms.Compose([
            transforms.Resize(224),   # Resize images to 224x224 as expected by ResNet-50
            transforms.ToTensor(),    # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
        ])

        self.dataset = datasets.CIFAR10(root='./data', transform=transform, download=True)
        # planes, cars and frogs
        index = random.choices([i for i ,e in enumerate(list(self.dataset.targets)) if e == 0], k=1000)
        index = np.append(index, random.choices([i for i ,e in enumerate(list(self.dataset.targets)) if e == 1], k=1000))
        index = np.append(index, random.choices([i for i ,e in enumerate(list(self.dataset.targets)) if e == 6], k=1000))
        print(len(index), 'total number of indices')
        data = torch.utils.data.Subset(self.dataset, index)
        self.dataset_train, self.dataset_val, self.dataset_test = torch.utils.data.random_split(data,[2500,500,0])
        # self.dataset_train, self.dataset_val, \
        #     self.dataset_test = split_dataset(self.dataset, True)
        # self.dataloader = DataLoader(self.dataset_train, batch_size=64, shuffle=True, num_workers=4)
        # for images, _ in self.dataloader:

        #     test_results = self.online_encoder(images)
        #     print(test_results, test_results.shape)

        self.online_projector = MLP(self.indim, projection_size, projection_hidden_size)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        self.use_momentum = True
        self.target_encoder = None
        self.target_projector = None
        moving_average_decay = 0.99
        self.target_ema_updater = EMA(moving_average_decay)
        
        DEFAULTAUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.RandomResizedCrop((self.image_size, self.image_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )
        self.augment1 = DEFAULTAUG
        self.augment2 = DEFAULTAUG

        self.device = 'cpu'
        if self.usecuda:
            self.cuda()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = device
            print('Using device:', device)

            #Additional Info when using cuda
            if self.device.type == 'cuda':
                print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    def initialize_target_network(self):
        """ give target network """
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p, q in zip(self.target_encoder.parameters(), self.target_projector.parameters()):
            p.requires_grad = False
            q.requires_grad = False

    def update_moving_average(self):
        " update target network by moving average "
        target_update_moving_average(self.target_ema_updater, self.online_encoder, \
                self.online_projector, self.target_encoder, self.target_projector)

    def forward(self, x, xt):
        """ forward BYOL """

        latent = self.online_encoder(x)

        z1_online = self.online_predictor(self.online_projector(latent))
        z2_online = self.online_predictor(self.online_projector(self.online_encoder(xt)))

        with torch.no_grad():
            z1_target = self.target_projector(self.target_encoder(xt))
            z2_target =  self.target_projector(self.target_encoder(x))

        byol_loss = self.compute_loss(z1_online, z1_target.detach()) + \
            self.compute_loss(z2_online, z2_target.detach()) # to symmetrize the loss

        return latent, byol_loss

    def compute_loss(self, z1, z2):
        """ loss for BYOL """
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)

        return torch.mean(2 - 2 * (z1 * z2.detach()).sum(dim=-1))

    def train_model(self):
        """ train model """
        # world_size = torch.cuda.device_count()
        # rank = (world_size, )
        # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        # Training loop
        num_epochs = 1000
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        losses = []
        self.dataset_train = DataLoader(self.dataset_train, batch_size=64, shuffle=True,\
            num_workers=self.num_workers, pin_memory=True) # batch_size=64
        self.dataset_val= DataLoader(self.dataset_val, batch_size=64, shuffle=True, \
            num_workers=self.num_workers, pin_memory=True)
        for epoch in range(num_epochs):
            total_losstrain = 0.0
            total_lossval = 0.0
            self.train()
            self.initialize_target_network()
            for images, _ in self.dataset_train:
                images1 = self.augment1(images)
                images2 = self.augment2(images)
                if self.usecuda:
                    images1, images2 = images1.cuda(), images2.cuda()
                _, loss = self(images1, images2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() # Update the model parameters

                # Update momentum encoder
                self.update_moving_average()

                total_losstrain += loss.detach().item()

            avg_loss = total_losstrain / len(self.dataset_train)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}, training", flush=True)
            self.logger.info(f'{epoch}: byol loss={avg_loss}')
            losses.append(avg_loss)
            self.eval()
            with torch.no_grad():
                for images, _ in self.dataset_val:
                    images1 = self.augment1(images)
                    images2 = self.augment2(images)
                    if self.usecuda:
                        images1, images2 = images1.cuda(), images2.cuda()
                    _, loss = self(images1, images2)
                    total_lossval += loss.detach().item()

            avg_loss = total_lossval / len(self.dataset_val)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}, validation", flush=True)
            self.logger.info(f'{epoch}: byol loss={avg_loss}')
        np.save(self.outdir + '/byol_loss.npy', np.array(losses))
        torch.save(self.state_dict(), self.outdir + '/byol_modelstate_dict.pth')

    def get_latent(self):
        """ get representation from trained encoder """
        dataloader = DataLoader(self.dataset_val, batch_size=64, \
            shuffle=False, num_workers=4, pin_memory=True)
        print(dataloader, 'dataloader')
        latent_space = []
        self.eval()
        with torch.no_grad():
            for images, _ in dataloader:
                if self.usecuda:
                    images = images.cuda()
                latent = self.online_encoder(images)
                latent_space.append(latent.detach().cpu().numpy())
        np.save(self.outdir + 'latent.npy', np.vstack(latent_space))

class LinearClassifier(nn.Module):
    """ Linear classifier """
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """ linear layer forward """
        return self.fc(x)


def train_linear_classifier(byol_model, train_loader, test_loader, logger):
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, 'device inside linear classifier')
    # Assuming the feature dimension is 2048 for BYOL
    classifier = LinearClassifier(input_dim=byol_model.indim, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_lc = torch.optim.Adam(classifier.parameters(), lr=0.001)

    byol_model.to(device)

    def remap_labels(labels, label_mapping):
        return torch.tensor([label_mapping[label.item()] for label in labels], dtype=torch.long)
    label_mapping = {2: 0, 3: 1, 4: 2}

    # Train classifier
    for epoch in range(10):  # Number of epochs
        classifier.train()
        byol_model.eval()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            labels = remap_labels(labels, label_mapping).to(device)
            # Get representations from BYOL model
            with torch.no_grad():
                representations = byol_model.online_encoder(images).detach()
            # Train classifier on these representations
            outputs = classifier(representations)
            loss = criterion(outputs, labels)

            optimizer_lc.zero_grad()
            loss.backward()
            optimizer_lc.step()

        logger.info(f"Epoch {epoch+1}, Loss: {loss.detach().item()}")

    # Evaluate classifier
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = remap_labels(labels, label_mapping).to(device)
            representations = byol_model.online_encoder(images).detach()
            outputs = classifier(representations)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logger.info(f'Accuracy of the network on the {len(test_loader)} test images: {100 * correct / total} %')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    prog="mcdevol",
    description="BYOL for image net",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    usage="%(prog)s --outdir [options]",
    add_help=False,
    )
    parser.add_argument("--outdir", type=str, \
        help="output directory", required=True)
    parser.add_argument("--cuda", \
        help="use GPU to train & cluster [False]", action="store_true")

    args = parser.parse_args()
    args.outdir = os.path.join(args.outdir, '')
    print(args.outdir ,'output directory')

    try:
        if not os.path.exists(args.outdir):
            print('create output folder')
            os.makedirs(args.outdir)
    except RuntimeError as e:
        print(f'output directory already exist. Using it {e}')

    logging.basicConfig(format='%(asctime)s - %(message)s', \
    level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S',
    filename=args.outdir + '/byol_resnet_loadbyolmodel_onlylc.log', filemode='w')
    args.logger = logging.getLogger()
    byol_model = BYOLmodel(args)

    byol_model.load_state_dict(torch.load(args.outdir +'/byol_modelstate_dict.pth',map_location=torch.device('cpu')), strict=False)
    # byol_model.train_model()
    # byol_model.get_latent()

    # Call the function
    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize(224),   # Resize images to 224x224 as expected by ResNet-50
        transforms.ToTensor(),    # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
    ])

    dataset = datasets.CIFAR10(root='./data', transform=transform, download=True)
    
    index = random.choices([i for i ,e in enumerate(list(dataset.targets)) if e == 2], k=1000)
    index = np.append(index, random.choices([i for i ,e in enumerate(list(dataset.targets)) if e == 3], k=1000))
    index = np.append(index, random.choices([i for i ,e in enumerate(list(dataset.targets)) if e == 4], k=1000))
    data = torch.utils.data.Subset(dataset, index)
    dataset_train, dataset_test = torch.utils.data.random_split(data,[2500,500])
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True,\
            num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_test, \
            num_workers=4, pin_memory=True)
    train_linear_classifier(byol_model, train_loader, test_loader, args.logger)
    args.logger.info("%s","done")
