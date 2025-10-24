#!/usr/bin/env python3

from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from random import Random

import numpy as np
from PIL import Image
from imageio.v3 import imread
from skimage.draw import rectangle_perimeter
from skimage.color import rgb2gray
from skimage.measure import label as skimage_label, regionprops
from skimage.morphology import flood
from skimage.util import invert

RNG = Random(8675309)

Line = tuple[tuple[float, float], tuple[float, float]]

STATE = {
    'filepath': Path(),
    'step': 0,
    'time': datetime.now(),
}

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
    check_time(f'saved {filename}')
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
    """Identify character and border (and artifact) regions."""
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
    """Create a visualization of different region components."""
    array = np.zeros(labels.shape)
    array[np.isin(labels, list(regions.keys()))] = 1
    save_image((array * 255).astype(np.uint8))


def hash_grid_radius_offsets(max_radius):
    """Generate the offsets for each radius away."""
    yield 0, [(0, 0)]
    for radius in range(1, max_radius):
        radius_keys = [
            (-radius, -radius),
            (-radius, radius),
            (radius, -radius),
            (radius, radius),
        ]
        for dim1_diff in range(-radius + 1, radius):
            for dim2_diff in (-radius, radius):
                radius_keys.append((dim1_diff, dim2_diff))
                radius_keys.append((dim2_diff, dim1_diff))
        yield radius, radius_keys


def k_nearest_neighbors(regions, k, grid_size):
    """Find the k nearest neighbors for each region.

    This implementation of kNN uses a hash grid to avoid unnecessary distance
    calculations. A hash grid assigns every point to a larger grid cell - for
    example, if the grid size is 10, the points (5, 13) and (6, 16) would both
    be assigned to grid cell (0, 10), while (57, 34) would be assigned to (50,
    30). To find the k nearest neighbors for a point, start by only checking
    distances to other points in the same grid cell, then to points one cell
    away, then two away, etc., until k neighbors are found. This avoids the need
    to check against points far away, in turn meaning that it is more efficient
    than the naive O(n^2) approach of checking every point against every other
    point. Empirically, for ~1300 points, this implementation is ~30% faster
    than the naive algorithm.

    Because grid cells are square, and because a point can be anywhere within a
    cell, care must be taken when determining whether neighbors are actually the
    nearest ones. (1, 1) and (19, 19) are one cell away (squared distance: 648),
    but are further apart than (1, 1) and (1, 21) despite the latter being two
    cells away (squared distance: 400). More generally, cells that are a radius
    `r` away could contain points `(r-1)*grid_size` to `sqrt(2)*r*grid_size`
    apart. The algorithm therefore only considers points nearer than the lower
    bound as confirmed, while keeping track of the further away points to be
    added later.
    """
    # initialize the hash grid by putting each region in the appropriate grid cell
    keys = []
    hash_grid = defaultdict(list)
    for region in regions:
        key = (region.centroid[0] // grid_size, region.centroid[1] // grid_size)
        keys.append(key)
        hash_grid[key].append(region)
    # initialize result variables
    distance_cache = {}
    all_nearest_neighbors = {}
    # loop over each region to look for its nearest neighbors
    for this_key, this_region in zip(keys, regions):
        this_centroid = this_region.centroid
        away_neighbors = []
        near_neighbors = []
        for radius, offsets in hash_grid_radius_offsets(len(hash_grid)):
            # pre-calculate the maximum distance we will consider, accounting for grid squareness
            max_distance = radius * radius * grid_size * grid_size
            # loop over the grid cells in the larger radius
            for offset in offsets:
                that_key = (this_key[0] + offset[0], this_key[1] + offset[1])
                # loop over the regions in that grid cell
                for that_region in hash_grid[that_key]:
                    # skip over the region itself
                    if that_region.label == this_region.label:
                        continue
                    that_centroid = that_region.centroid
                    # retrieve or calculate the distance between regions
                    distance_key = tuple(sorted([this_centroid, that_centroid]))
                    if distance_key not in distance_cache:
                        dx = this_centroid[0] - that_centroid[0]
                        dy = this_centroid[1] - that_centroid[1]
                        distance_cache[distance_key] = dx * dx + dy * dy
                    distance = distance_cache[distance_key]
                    away_neighbors.append((distance, that_region.label))
            # add regions that were too far away but are now eligible
            new_away_neighbors = []
            for distance, that_label in away_neighbors:
                if distance <= max_distance:
                    near_neighbors.append((distance, that_label))
                else:
                    new_away_neighbors.append((distance, that_label))
            away_neighbors = new_away_neighbors
            # if there are enough near neighbors, this region is done
            if len(near_neighbors) >= k:
                break
        all_nearest_neighbors[this_label] = [
            that_label for _, that_region in sorted(near_neighbors)[:k]
        ]
        result = all_nearest_neighbors[this_label]
        assert len(result) == len(set(result))
    # return the list of nearest neighbors
    return all_nearest_neighbors


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


def visualize_components(regions, labels, components):
    array = np.zeros((*labels.shape, 3)).astype(np.uint8)
    for component in components:
        rgb = (
            RNG.randrange(128, 255),
            RNG.randrange(128, 255),
            RNG.randrange(128, 255),
        )
        array_min_row, array_min_col = array.shape[:2]
        array_max_row = 0
        array_max_col = 0
        for region_index in component:
            region = regions[region_index]
            array[labels == region.label] = np.repeat(
                [rgb],
                array[labels == region.label].shape[0],
                axis=0,
            )
            min_row, min_col, max_row, max_col = region.bbox
            array_min_row = min(min_row, array_min_row)
            array_min_col = min(min_col, array_min_col)
            array_max_row = max(max_row, array_max_row)
            array_max_col = max(max_col, array_max_col)
        perimeter_mask = rectangle_perimeter(
            start=(array_min_row, array_min_col),
            end=(array_max_row, array_max_col),
            shape=array.shape,
            clip=True,
        )
        array[perimeter_mask] = rgb
    save_image(array)


def pipeline(path):
    STATE['filepath'] = path
    # read the image
    array = imread(path)
    save_image(array)
    # convert to black-and-white
    array = (rgb2gray(array) * 255 > 127) * np.ones(array.shape[:2])
    array = (array * 255).astype(np.uint8)
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
    nearest_neighbors = k_nearest_neighbors(
        character_regions.values(),
        1,
        min(array.shape[0], array.shape[1]) // 20,
    )
    components = find_connected_components(nearest_neighbors)
    visualize_components(character_regions, labels, components)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument('images', metavar='image', type=Path, nargs='+')
    args = arg_parser.parse_args()
    args.images = sorted(set(path.expanduser().resolve() for path in args.images))
    for image_path in args.images:
        pipeline(image_path)


if __name__ == '__main__':
    main()
