import argparse
import os

import numpy as np
import torch

from PIL import Image
from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks, plot_img_and_mask

def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_dense_crf=False,
                use_gpu=False):

    net.eval()
    img_height, img_width = full_img.size[1], full_img.size[0]

    img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(img)

    left_square, right_square = split_img_into_squares(img)

    # Convert PIL Image -> NumPy array
    if isinstance(left_square, Image.Image):
        left_square = np.array(left_square, dtype=np.float32)
    if isinstance(right_square, Image.Image):
        right_square = np.array(right_square, dtype=np.float32)

    # HWC -> CHW and add batch dimension
    X_left = torch.tensor(left_square.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
    X_right = torch.tensor(right_square.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)

    if use_gpu:
        X_left = X_left.cuda()
        X_right = X_right.cuda()

    with torch.no_grad():
        output_left = net(X_left)
        output_right = net(X_right)
    
        # Squeeze batch dimension and channel dimension
        left_probs = output_left.squeeze().cpu().numpy()
        right_probs = output_right.squeeze().cpu().numpy()
        
        if left_probs.ndim == 3:  # (1, H, W)
            left_probs = left_probs[0]
        if right_probs.ndim == 3:
            right_probs = right_probs[0]

        left_probs = left_probs.astype(np.float32)
        right_probs = right_probs.astype(np.float32)

    # Merge left and right halves directly
    full_mask = merge_masks(left_probs, right_probs, img_width)
    
    if use_dense_crf:
        from utils import dense_crf
        full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)
    
    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE', help="Model file path")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='Input image filenames', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+',
                        help='Output mask filenames')
    parser.add_argument('--cpu', '-c', action='store_true', default=False,
                        help="Do not use GPU")
    parser.add_argument('--viz', '-v', action='store_true', default=False,
                        help="Visualize results")
    parser.add_argument('--no-save', '-n', action='store_true', default=False,
                        help="Do not save masks")
    parser.add_argument('--no-crf', '-r', action='store_true', default=False,
                        help="Do not apply dense CRF")
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help="Threshold for mask pixels")
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help="Scaling factor for input images")
    parser.add_argument('--channel-txt', '-ct', default='model_channels.txt',
                        help="Path to file containing channel info")
    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append(f"{pathsplit[0]}_OUT{pathsplit[1]}")
    elif len(in_files) != len(args.output):
        print("Error: Input and output files length mismatch")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    # Initialize UNet
    net = UNet(n_channels=3, n_classes=1, f_channels=args.channel_txt)

    print(f"Loading model {args.model}")
    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be slow")
    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print(f"\nPredicting image {fn} ...")
        img = Image.open(fn)
        if img.size[0] < img.size[1]:
            print("Warning: image height larger than width")

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf=not args.no_crf,
                           use_gpu=not args.cpu)

        if args.viz:
            print(f"Visualizing results for image {fn}, close to continue ...")
            plot_img_and_mask(img, mask)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_fn)
            print(f"Mask saved to {out_fn}")
