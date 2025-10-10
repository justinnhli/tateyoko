#!/usr/bin/env python3


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
}

def save_image(array, mode=None):
    image = Image.fromarray(array, mode=mode)
    image.save(f'step{STATE["step"]:02d}.png')
    STATE['step'] += 1


def create_crop_mask(array):
    # type: (np.ndarray) -> np.ndarray
    """Crop to the contents of the page."""
    # assume the topleft pixel is the background and flood it
    flood_mask = flood(array, (0, 0))
    # label all regions that were not flooded (label() considers 0 as background by default)
    flooded = (invert(flood_mask) * np.ones(array.shape) * 255).astype('uint8')
    labels = label(flooded)
    # return a mask of the largest region
    largest_region = max(
        regionprops(labels),
        key=(lambda region: region.area),
    )
    return labels == largest_region.label


def crop(array, mask):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    rows, columns = np.where(mask)
    row1 = min(rows)
    row2 = max(rows)
    col1 = min(columns)
    col2 = max(columns)
    return array[row1:row2, col1:col2]


def main():
    # read the image
    array = imread('Ranking-Bunjin suikoden.jpg')
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
    crop_mask = create_crop_mask(array)
    array = crop(array, crop_mask)
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
    for region in border_regions:
        array[labels == region.label] = 1
    array = (array * 255).astype('uint8')
    save_image(array)
    array = np.zeros(array.shape)
    for region in character_regions:
        array[labels == region.label] = 1
    array = (array * 255).astype('uint8')
    save_image(array)


if __name__ == '__main__':
    main()
