import argparse
import os
import numpy as np
import torch
from PIL import Image

from unet import UNet
from utils import resize_and_crop, normalize, plot_img_and_mask


def predict_img(net, full_img, scale_factor=0.5, threshold=0.5, use_gpu=False):
    net.eval()

    # SAME preprocessing as training
    img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(img)

    # HWC -> CHW
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)  # (1, 3, H, W)

    img = torch.tensor(img, dtype=torch.float32)

    if use_gpu:
        img = img.cuda()

    with torch.no_grad():
        pred = net(img)          # (1, 1, H, W)
        pred = pred.squeeze()    # (H, W)
        pred = pred.cpu().numpy()

    mask = (pred > threshold).astype(np.uint8)
    return mask


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth')
    parser.add_argument('--input', '-i', nargs='+', required=True)
    parser.add_argument('--output', '-o', nargs='+')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--mask-threshold', type=float, default=0.5)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--channel-txt', default='model_channels.txt')
    return parser.parse_args()


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()

    net = UNet(
        n_channels=3,
        n_classes=1,
        f_channels=args.channel_txt
    )

    print(f"Loading model {args.model}")

    if not args.cpu:
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.load_state_dict(torch.load(args.model, map_location='cpu'))

    print("Model loaded!")

    for fn in args.input:
        print(f"\nPredicting image {fn} ...")

        img = Image.open(fn).convert("RGB")

        mask = predict_img(
            net=net,
            full_img=img,
            scale_factor=args.scale,
            threshold=args.mask_threshold,
            use_gpu=not args.cpu
        )

        if args.viz:
            plot_img_and_mask(img, mask)

        if not args.no_save:
            out_fn = os.path.splitext(fn)[0] + "_OUT.png"
            mask_to_image(mask).save(out_fn)
            print(f"Mask saved to {out_fn}")
