#!/usr/bin/env python3

from argparse import ArgumentParser
from itertools import combinations
from collections import defaultdict
from datetime import datetime
from math import inf as INF
from pathlib import Path
from random import Random

import numpy as np
from PIL import Image
from imageio.v3 import imread
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.morphology import flood
from skimage.util import invert

RNG = Random(8675309)

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


def save_image(array):
    image = Image.fromarray(array)
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


def identify_characters_borders(array):
    character_regions = {}
    border_regions = {}
    min_dimension = min(array.shape[0], array.shape[1]) // 100
    max_dimension = min(array.shape[0], array.shape[1]) // 20
    labels = label(array)
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
            character_regions[region.label] = region
        else:
            border_regions[region.label] = region
    return labels, character_regions, border_regions


def visualize_regions(labels, regions):
    array = np.zeros(labels.shape)
    array[np.isin(labels, list(regions.keys()))] = 1
    save_image((array * 255).astype('uint8'))


def k_nearest_neighbors(regions, k):
    # calculate distances between all regions
    distances = defaultdict(list)
    for region1, region2 in combinations(regions, 2):
        i = region1.label
        j = region2.label
        centroid1 = region1.centroid
        centroid2 = region2.centroid
        dx = centroid1[0] - centroid2[0]
        dy = centroid1[1] - centroid2[1]
        distance = dx * dx + dy * dy
        distances[i].append((distance, j))
        distances[j].append((distance, i))
    # get the k nearest neighbors
    neighbors = {}
    for this_label, region_distances in distances.items():
        neighbors[this_label] = [
            pair[1] for pair in sorted(region_distances)[:k]
        ]
    return neighbors


def find(union_find, i):
    path = set()
    rep = i
    while union_find[rep] != rep:
        path.add(rep)
        rep = union_find[rep]
    for node in path:
        union_find[node] = rep
    return rep


def union(union_find, i, j):
    path = set()
    rep = i
    while union_find[rep] != rep:
        path.add(rep)
        rep = union_find[rep]
    path.add(rep)
    rep = j
    while union_find[rep] != rep:
        path.add(rep)
        rep = union_find[rep]
    path.add(rep)
    for node in path:
        union_find[node] = rep
    return rep


def find_connected_components(neighbors):
    # type: (Mapping[int, Region]) -> list[set[int]]
    # use union-find to identify connected components
    union_find = {label: label for label in neighbors}
    for label, nearest_neighbors in neighbors.items():
        for neighbor in nearest_neighbors:
            union(union_find, label, neighbor)
    # extract out the connected components
    components = defaultdict(set)
    for label in neighbors:
        rep = find(union_find, label)
        components[rep].add(label)
    return list(components.values())


def pipeline(path):
    # read the image
    array = imread(path)
    save_image(array)
    # convert to black-and-white
    array = (rgb2gray(array) * 255 > 127) * np.ones(array.shape[:2])
    array = (array * 255).astype('uint8')
    save_image(array)
    # crop to just the page
    array = crop(array)
    save_image(array)
    # invert the image colors
    array = invert(array)
    save_image(array)
    # separate characters from borders
    labels, character_regions, border_regions = identify_characters_borders(array)
    visualize_regions(labels, border_regions)
    visualize_regions(labels, character_regions)
    # find nearest neighbors and visualize
    components = find_connected_components(k_nearest_neighbors(character_regions.values(), 1))
    array = np.zeros((*array.shape, 3)).astype('uint8')
    for component in components:
        rgb = (
            RNG.randrange(128, 255),
            RNG.randrange(128, 255),
            RNG.randrange(128, 255),
        )
        for region_index in component:
            region = character_regions[region_index]
            array[labels == region.label] = np.repeat(
                [rgb],
                array[labels == region.label].shape[0],
                axis=0,
            )
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
