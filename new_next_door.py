#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from random import Random

import numpy as np
from PIL import Image
from imageio.v3 import imread
from skimage.draw import line, rectangle_perimeter
from skimage.color import rgb2gray
from skimage.measure import label as skimage_label, regionprops
from skimage.morphology import flood
from skimage.util import invert
from scipy import signal

from subvalley import extract_subvalleys_at_intersections, report_subvalleys

RNG = Random(8675309)

Line = tuple[tuple[float, float], tuple[float, float]]

STATE = {
    'filepath': Path(),
    'step': 0,
    'time': datetime.now(),
}


def reset_state():
    STATE['filepath'] = Path()
    STATE['step'] = 0
    STATE['time'] = datetime.now()


def check_time(message=''):
    """Print out the current and elapsed time."""
    prev_time = STATE['time']
    curr_time = datetime.now()
    if message:
        print(f'{prev_time.isoformat()} (+{(curr_time - prev_time).seconds}); {message}')
    else:
        print(f'{prev_time.isoformat()} (+{(curr_time - prev_time).seconds})')
    STATE['time'] = curr_time


def save_image(array):
    """Save the array as an image, with an auto-incremented filename."""
    image = Image.fromarray(array)
    filename = f'{STATE["filepath"].stem}-step{STATE["step"]:02d}.png'
    image.save(filename)
    STATE['step'] += 1


def crop(array):
    """Crop to the contents of the page."""
    # assume the topleft pixel is the background and flood it
    flood_mask = flood(array, (0, 0))
    # label all regions that were not flooded (skimage_label() considers 0 as background by default)
    flooded = (invert(flood_mask) * np.ones(array.shape) * 255).astype(np.uint8)
    labels = skimage_label(flooded)
    # find the largest region
    largest_region = max(
        regionprops(labels),
        key=(lambda region: region.area),
    )
    # crop
    min_row, min_col, max_row, max_col = largest_region.bbox
    return array[min_row:max_row, min_col:max_col]


def identify_characters_borders(array):
    """Identify character (or object) and border regions."""
    character_regions = {}
    border_regions = {}
    min_dimension = min(array.shape[0], array.shape[1]) // 100
    max_dimension = min(array.shape[0], array.shape[1]) // 20
    labels = skimage_label(array)
    for region in regionprops(labels):
        min_row, min_col, max_row, max_col = region.bbox
        width = max_col - min_col # the width of the region 
        height = max_row - min_row # the height of the region 
        if width < min_dimension and height < min_dimension:
            # discard small image features 
            continue
        is_character = (
            width < max_dimension # width less than 1/20 of the image 
            and height < max_dimension # height less than 1/20 of the image 
            and (width / height) < 10 # width: height ratio less than 10 
            and (height / width) < 10 # height: width ratio of less than 10 
        )
        if is_character:
            character_regions[region.label] = region
        else:
            border_regions[region.label] = region
    return labels, character_regions, border_regions


def visualize_regions(labels, regions):
    """Create a visualization of different region components."""
    array = np.zeros(labels.shape)
    array[np.isin(labels, list(regions.keys()))] = 1
    save_image((array * 255).astype(np.uint8))
    return array


def sum_dimension(array, dimension):
    """Sum an array along the rows or the columns."""
    if dimension == 'row':
        return np.sum(array, axis=1)
    elif dimension == 'col':
        return np.sum(array, axis=0)
    else:
        raise ValueError(' '.join([
            'dimension should be either "row" or "col"',
            f'but got "{dimension}"',
        ]))


def find_peaks_and_valleys(signal_data, min_distance=10):
    """Find peaks (high values) and valleys (low values) in the signal."""
    peaks, peak_properties = signal.find_peaks(signal_data, distance=min_distance)
    valleys, valley_properties = signal.find_peaks(-signal_data, distance=min_distance)
    return peaks, valleys


def refine_gaps_from_peaks_and_valleys(row_pixel_counts, col_pixel_counts, peak_rows, valley_rows, peak_cols, valley_cols,
                                        white_threshold=0.3):
    """ Refine peaks/valleys to find ACTUAL gaps (true separations between text).
        - for each detected valley, check if it's actually a gap using a threshold.
        - only keep valleys where white_ratio <= white_threshold.
    """
    row_white_ratios = row_pixel_counts / len(col_pixel_counts)
    col_white_ratios = col_pixel_counts / len(row_pixel_counts)
    
    refined_valley_rows = set()
    for row_idx in valley_rows:
        if row_white_ratios[row_idx] <= white_threshold:
            refined_valley_rows.add(row_idx)
    
    refined_valley_cols = set()
    for col_idx in valley_cols:
        if col_white_ratios[col_idx] <= white_threshold:
            refined_valley_cols.add(col_idx)
    
    return refined_valley_rows, refined_valley_cols


def visualize_non_characters(image, rows, cols):
    # create an empty (all black) image
    array = np.zeros(image.shape)
    row_mask = np.isin(np.arange(array.shape[0]), list(rows))
    # make the rows white 
    array = np.ma.masked_array(
        array,
        np.repeat(row_mask, array.shape[1]).reshape(array.shape),
        fill_value=1,
    ).filled()
    # make the columns white 
    col_mask = np.isin(np.arange(array.shape[1]), list(cols))
    array = np.ma.masked_array(
        array,
        (
            np.repeat(col_mask, array.shape[0])
            .reshape(array.transpose().shape)
            .transpose()
        ),
        fill_value=1,
    ).filled()
    # save the image
    save_image((array * 255).astype(np.uint8))
    return array


def pipeline(path, k):
    reset_state()
    STATE['filepath'] = path
    
    # read the image
    array = imread(path)
    check_time(f'read in the image')
    save_image(array)
    
    # convert to black-and-white
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    elif array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    array = (rgb2gray(array) * 255 > 127) * np.ones(array.shape[:2])
    array = (array * 255).astype(np.uint8)
    
    # crop to just the page
    array = crop(array)
    check_time(f'cropped the image')
    save_image(array)
    
    # separate characters from borders
    array = invert(array)
    labels, character_regions, border_regions = identify_characters_borders(array)
    check_time(f'separated characters from borders')
    visualize_regions(labels, border_regions)
    character_image = visualize_regions(labels, character_regions)
    check_time(f'visualized characters and borders')
    
    # sum white pixels along rows and columns
    row_pixel_counts = sum_dimension(character_image, 'row')
    col_pixel_counts = sum_dimension(character_image, 'col')
    
    # find peaks and valleys
    min_distance = min(array.shape[0], array.shape[1]) // 50
    peak_rows, valley_rows = find_peaks_and_valleys(row_pixel_counts, min_distance)
    peak_cols, valley_cols = find_peaks_and_valleys(col_pixel_counts, min_distance)
    check_time(f'found peaks and valleys')
    
    # refine valleys to find actual gaps
    gap_rows, gap_cols = refine_gaps_from_peaks_and_valleys(
        row_pixel_counts, col_pixel_counts,
        peak_rows, valley_rows, peak_cols, valley_cols,
        white_threshold=0.3
    )
    check_time(f'refined gaps from peaks and valleys')
    
    # visualize the refined gaps
    visualize_non_characters(character_image, gap_rows, gap_cols)
    check_time(f'visualized refined gaps')
    
    # find sub-valleys at gap intersections
    subvalley_info = extract_subvalleys_at_intersections(
        binary_array=array,
        gap_rows=gap_rows,
        gap_cols=gap_cols,
        window_size=20
    )
    check_time(f'extracted sub-valleys at intersections')
    
    # report sub-valleys
    report_subvalleys(subvalley_info['subvalleys'])
    
    # store subvalley data for later use
    subvalleys = subvalley_info['subvalleys']
    all_subvalley_coords = subvalley_info['all_coordinates']
    
    print(f"\nStored {len(subvalleys)} sub-valleys with {len(all_subvalley_coords)} total black pixels")
    print(f"Sub-valley data ready for custom grouping logic")


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('images', metavar='image', type=Path, nargs='+')
    arg_parser.add_argument('-k', default=3, type=int)
    args = arg_parser.parse_args()
    args.images = sorted(set(path.expanduser().resolve() for path in args.images))
    for image_path in args.images:
        pipeline(image_path, k=args.k)


if __name__ == '__main__':
    main()