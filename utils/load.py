#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    ids = []
    for f in os.listdir(dir):
        # Skip hidden files and folders
        if f.startswith("."):
            continue

        full_path = os.path.join(dir, f)

        # Skip directories
        if not os.path.isfile(full_path):
            continue

        # Keep only image files
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            ids.append(os.path.splitext(f)[0])

    return ids


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return [(id, i) for id in ids for i in range(n)]


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)

def to_cropped_imgs(ids, dir, suffix, scale):
    """Load and resize full images (no cropping)."""
    for id in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield im

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.tif', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.tif', scale)

    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.tif')
    mask = Image.open(dir_mask + id + '_mask.tif')
    return np.array(im), np.array(mask)
