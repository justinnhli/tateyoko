#!/usr/bin/env python3


import numpy as np
from PIL import Image
from PIL.ImageDraw import Draw
from imageio import imread
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.draw import line as line_pixels
from skimage.feature import canny
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import erosion, dilation, flood, flood_fill
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
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
    region_props = regionprops(labels)
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


def hough_test_angles():
    # type: () -> np.ndarray
    return np.concatenate(
        (
            # vertical lines
            np.linspace(-np.pi / 72, np.pi / 72, 360, endpoint=False),
            # horizontal lines
            np.linspace(-np.pi / 72, np.pi / 72, 360, endpoint=False) + np.pi / 2,
        ),
    )


def hough_transform(array):
    # type: (np.ndarray) -> tuple[Line, ...]
    """Apply Hough transform to identify straight lines."""
    h, theta, d = hough_line(
        array,
        theta=hough_test_angles(),
    )
    result = []
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        (x, y) = dist * np.array([np.cos(angle), np.sin(angle)])
        slope = np.tan(angle + np.pi / 2)
        y_intercept = y - (x * slope)
        if abs(slope) < 1:
            # horizontal line
            result.append((
                (0, y_intercept),
                (array.shape[0], y_intercept + (array.shape[0] * slope)),
            ))
        else:
            # vertical line
            result.append((
                (x - 100, ((x - 100) * slope) + y_intercept),
                (x + 100, ((x + 100) * slope) + y_intercept),
            ))
    return tuple(result)


def probabilistic_hough_transform(array):
    # type: (np.ndarray) -> tuple[Line, ...]
    """Apply probabilistic Hough transform to identify straight lines."""
    lines = probabilistic_hough_line(
        array,
        theta=hough_test_angles(),
        line_length=min(array.shape) // 4,
        #line_gap=min(array.shape) // 20,
    )
    return tuple(((x1, y1), (x2, y2)) for ((x1, x2), (y1, y2)) in lines)


def visualize_lines(array, lines):
    # type: (np.ndarray, Sequence[Line]) -> Image
    image = Image.fromarray(array).convert('RGB')
    draw = Draw(image, 'RGB')
    for (x1, y1), (x2, y2) in lines:
        draw.line(
            ((x1, y1), (x2, y2)),
            fill=(255, 0, 0),
            width=1,
        )
    return image


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
    '''
    # erode to remove artifacts
    array = erosion(array)
    save_image(array)
    '''
    # crop to just the page
    crop_mask = create_crop_mask(array)
    array = crop(array, crop_mask)
    clean_array = array
    save_image(array, 'L')
    # invert the image colors
    array = invert(array)
    save_image(array)
    # remove small regions that are not part of the grid
    labels = label(array)
    region_props = regionprops(labels)
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
