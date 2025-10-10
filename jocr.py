#!/usr/bin/env python3

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from imageio import imread
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import flood
from skimage.util import invert


Line = tuple[tuple[float, float], tuple[float, float]]

STATE = {
    'step': 0,
    'time': datetime.now(),
}

def check_time(message=''):
    prev_time = STATE['time']
    curr_time = datetime.now()
    if message:
        print(f'{prev_time.isoformat()} (+{(curr_time - prev_time).seconds}); {message}')
    else:
        print(f'{prev_time.isoformat()} (+{(curr_time - prev_time).seconds})')
    STATE['time'] = curr_time


def save_image(array, mode=None):
    image = Image.fromarray(array, mode=mode)
    filename = f'step{STATE["step"]:02d}.png'
    image.save(filename)
    check_time(f'saved {filename}')
    STATE['step'] += 1


def crop(array):
    # type: (np.ndarray) -> np.ndarray
    """Crop to the contents of the page."""
    # assume the topleft pixel is the background and flood it
    flood_mask = flood(array, (0, 0))
    # label all regions that were not flooded (label() considers 0 as background by default)
    flooded = (invert(flood_mask) * np.ones(array.shape) * 255).astype('uint8')
    labels = label(flooded)
    # find the largest region
    largest_region = max(
        regionprops(labels),
        key=(lambda region: region.area),
    )
    # crop
    min_row, min_col, max_row, max_col = largest_region.bbox
    return array[min_row:max_row, min_col:max_col]


def pipeline(path):
    # read the image
    array = imread(path)
    save_image(array)
    # convert to grayscale
    array = rgb2gray(array)
    array = (array * 255).astype('uint8')
    save_image(array, 'L')
    # threshold to black-and-white
    threshold = 127
    array = (array > 127) * np.ones(array.shape)
    array = (array * 255).astype('uint8')
    save_image(array, 'L')
    # crop to just the page
    array = crop(array)
    save_image(array, 'L')
    # invert the image colors
    array = invert(array)
    save_image(array)
    # separate characters from borders
    labels = label(array)
    array = np.zeros(array.shape)
    character_regions = []
    border_regions = []
    min_dimension = min(array.shape[0], array.shape[1]) // 100
    max_dimension = min(array.shape[0], array.shape[1]) // 20
    for region in regionprops(labels):
        min_row, min_col, max_row, max_col = region.bbox
        width = max_col - min_col # the width of the region
        height = max_row - min_row # the height of the region
        if width < min_dimension and height < min_dimension:
            # discard small image artifacts
            continue
        is_character = (
            width < max_dimension # width less than 1/20 of the image
            and height < max_dimension # height less than 1/20 of the image
            and (width / height) < 10 # width:height ratio less than 10
            and (height / width) < 10 # height:width ratio less than 10
        )
        if is_character:
            character_regions.append(region)
        else:
            border_regions.append(region)
    array = np.zeros(array.shape)
    array[np.isin(labels, [region.label for region in border_regions])] = 1
    array = (array * 255).astype('uint8')
    save_image(array)
    array = np.zeros(array.shape)
    array[np.isin(labels, [region.label for region in character_regions])] = 1
    array = (array * 255).astype('uint8')
    save_image(array)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('images', metavar='image', type=Path, nargs='+')
    args = arg_parser.parse_args()
    args.images = sorted(set(path.expanduser().resolve() for path in args.images))
    for image_path in args.images:
        pipeline(image_path)


if __name__ == '__main__':
    main()
