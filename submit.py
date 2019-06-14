import os
from PIL import Image

import torch
from  argparse import ArgumentParser
from tqdm import tqdm

from predict import predict_img
from utils import rle_encode
from unet import UNet


def submit(net, args, gpu=False):
    """Used for Kaggle submission: predicts and encode all test images"""
    dir = os.path.join(args.dataset_folder, 'data/test/')

    N = len(list(os.listdir(dir)))

    rle_encoded_masks = []
    for index, i in enumerate(tqdm(os.listdir(dir))):
        img = Image.open(dir + i)

        mask = predict_img(net, img, gpu, use_dense_crf=not args.no_crf)
        enc = rle_encode(mask)
        rle_encoded_masks.append(enc)

    with open('submission.csv', 'w') as f:
        f.write('img,rle_mask\n')
        for index, i in enumerate(tqdm(os.listdir(dir))):
            print('{}/{}'.format(index, N))
            f.write('{},{}\n'.format(i, ' '.join(map(str, rle_encoded_masks[index]))))


if __name__ == '__main__':
    parser = ArgumentParser(description='U-Net training script')
    parser.add_argument('--dataset_folder', type=str, required=True, help='Path to dataset folder')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint')
    parser.add_argument('--no_crf', default=False, action='store_true', help="Do not use dense CRF postprocessing")
    args = parser.parse_args()

    net = UNet(3, 1)
    net.load_state_dict(torch.load(args.checkpoint))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device)


    submit(net, args, True)
