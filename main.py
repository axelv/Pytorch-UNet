# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
import joblib
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import random_split, Sampler, WeightedRandomSampler
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm

from unet_small import UNetSmall

plt.ion()  # interactive mode


def weighted_mse_loss(input, target, weight):
    return torch.mean(weight.view(-1, 1, 1, 1) * (input - target) ** 2)


def init(data_dir='./data', val_percent=0.1, batch_size=10):
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.25])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.25])
        ]),
    }
    train_dataset = datasets.FashionMNIST("./data",
                                          train=True,
                                          transform=data_transforms['train'],
                                          target_transform=None,
                                          download=True)

    test_dataset = datasets.FashionMNIST("./data",
                                         train=False,
                                         transform=data_transforms['train'],
                                         target_transform=None,
                                         download=True)

    N_val = len(test_dataset)
    N_train = len(train_dataset)
    dataset_list = [train_dataset, test_dataset]

    image_datasets = {x: dataset for x, dataset in zip(['train', 'val'], dataset_list)}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return dataloaders, device


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, eval_metric, optimizer, scheduler, dataloaders, device, num_epochs=25, discard_label=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_mse = 0.0

    sample_mse = list()
    sample_labels = list()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            iterator = tqdm(dataloaders[phase], leave=False)
            for inputs, labels in iterator:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    weights = (1 - labels == torch.ones_like(labels)*discard_label).float()
                    loss = criterion(outputs, inputs, weights)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                iterator.set_description("Loss: %.3f" % loss.item())

                if phase == "val":
                    sample_mse.append(eval_metric(outputs, inputs).to("cpu").numpy())
                    sample_labels.append(labels.to("cpu").numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss > best_mse:
                best_mse = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                joblib.dump({'mse': sample_mse, 'labels': sample_labels}, "validation_results.joblib")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val MSE: {:4f}'.format(best_mse))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(preds[j]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def main():
    dataloaders, device = init(batch_size=100)
    criterion = nn.MSELoss()

    def eval_metric(input: torch.Tensor, target: torch.Tensor):
        error = input - target
        error = error * error
        return error.mean(list(range(1, len(error.shape))))

    model = UNetSmall(n_channels=1)
    model = model.to(device)
    lr = 0.1
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    best_model = train_model(model, criterion=weighted_mse_loss, eval_metric=eval_metric, optimizer=optimizer,
                             scheduler=scheduler, dataloaders=dataloaders, device=device)


if __name__ == "__main__":
    main()
