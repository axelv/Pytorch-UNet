import sys
import os
from optparse import OptionParser
import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torchvision.transforms import transforms
from eval import eval_net
from unet_small import UNet
# from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5):

    # dir_img = 'data/train/'
    # dir_mask = 'data/train_masks/'
    dir_checkpoint = 'checkpoints/'
    #
    # ids = get_ids(dir_img)
    # ids = split_ids(ids)
    #
    # iddataset = split_train_val(ids, val_percent)

    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.ToTensor()
    ])

    dataset = FashionMNIST("./data", train=True, transform=transform, target_transform=None, download=True)

    N_val = int(round(len(dataset)*val_percent))
    N_train = len(dataset) - N_val
    trainset, testset = random_split(dataset, [N_train, N_val])
    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, N_train,
               N_val, str(save_cp), str(gpu)))

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.MSELoss()
    # scheduler = StepLR(optimizer, )

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # reset the generators
        # train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        # val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

        epoch_loss = 0

        dataloader = {'train': DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4),
                      'val': DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)}
        phase = 'train'
        print("Start Training")
        for i, (input_batch, target_batch) in tqdm(enumerate(dataloader['train']), leave=False):

            if phase == 'train':
                # scheduler.step()
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            # imgs = np.array([i[0] for i in b]).astype(np.float32)
            # true_masks = np.array([i[1] for i in b])

            # imgs = torch.from_numpy(imgs)
            # true_masks = torch.from_numpy(true_masks)

            # if gpu:
            #     imgs = imgs.cuda()
            #     true_masks = true_masks.cuda()

            pred_batch = net(input_batch)
            # masks_probs_flat = masks_pred.view(-1)

            # true_masks_flat = true_masks.view(-1)

            loss = criterion(input_batch, pred_batch)
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            input_batch, target_batch = next(dataloader['val'])
            net.eval()
            pred_batch = net(input_batch)
            loss = criterion(input_batch, pred_batch)
            print('Validation MSE: {}'.format(loss))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=1, n_classes=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
