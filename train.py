import sys
import os
from argparse import ArgumentParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch


def train_net(args, net, val_percent=0.05, save_cp=True):

    dir_img = os.path.join(args.dataset_folder, 'data/train/')
    dir_mask = os.path.join(args.dataset_folder, 'data/train_masks/')
    dir_checkpoint = os.path.join(args.dataset_folder,'checkpoints/')

    ids = get_ids(dir_img)
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
    '''.format(args.epochs, args.batch_size, args.lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()

    for epoch in range(args.epochs):
        print('Starting epoch {}/{}.'.format(args.epochs + 1, args.epochs))
        net.train()

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, args.img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, args.img_scale)

        epoch_loss = 0

        for i, b in enumerate(batch(train, args.batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            # if gpu:
            imgs = imgs.cuda()
            true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * args.batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            val_dice = eval_net(net, val)
            print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))


if __name__ == '__main__':
    parser = ArgumentParser(description='U-Net training script')
    parser.add_argument('--dataset_folder', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.1,  help='Learning rate')
    parser.add_argument('--pretrained', type=str, default='', help='Pretrained file model')
    parser.add_argument('--img_scale', type=float, default=0.5, help='Downscaling factor of the images')

    args = parser.parse_args()

    net = UNet(n_channels=3, n_classes=1)

    if args.pretrained:
        net.load_state_dict(torch.load(args.pretrained))
        print('Model loaded from {}'.format(args.pretrained))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device)

    train_net(args, net)
