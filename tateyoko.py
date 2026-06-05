#!/usr/bin/env python3

# pylint: disable = missing-function-docstring

from argparse import ArgumentParser
from json import load as open_json
from subprocess import run
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from imageio.v3 import imread
from skimage.draw import line, rectangle_perimeter
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import label as skimage_label, regionprops
from skimage.morphology import disk
from skimage.filters.rank import maximum as maximum_filter
from skimage.util import invert

RNG = Random(8675309)

Line = tuple[tuple[float, float], tuple[float, float]]
Color = tuple[int, int, int]
State = TypedDict('State', {'filepath': Path, 'step': int, 'time': datetime})

STATE = {
    'filepath': Path(),
    'step': 0,
    'time': datetime.now(),
} # type: State


def reset_state():
    # type: () -> None
    STATE['filepath'] = Path()
    STATE['step'] = 0
    STATE['time'] = datetime.now()


def check_time(message=''):
    # type: (str) -> None
    """Print out the current and elapsed time."""
    prev_time = STATE['time']
    curr_time = datetime.now()
    if message:
        print(f'{prev_time.isoformat()} (+{(curr_time - prev_time).seconds}); {message}')
    else:
        print(f'{prev_time.isoformat()} (+{(curr_time - prev_time).seconds})')
    STATE['time'] = curr_time


def save_image(array, path=None, filename=None):
    # type: (NDArray, Path, str) -> None
    """Save the array as an image, with an auto-incremented filename."""
    if path is None:
        path = Path()
    path = path.expanduser().resolve()
    if not path.exists():
        path.mkdir(parents=True)
    image = Image.fromarray(array)
    if filename is None:
        filename = f'{STATE["filepath"].stem}-step{STATE["step"]:02d}.png'
    elif not filename.endswith('.png'):
        filename += '.png'
    image.save(path / filename)
    STATE['step'] += 1


def crop(array):
    # type: (NDArray) -> tuple[tuple[int, int], NDArray]
    """Crop to the contents of the page."""
    # identify beige (the color of the paper) via HSV
    hsv_img = rgb2hsv(array)
    hue_img = hsv_img[:, :, 0]
    sat_img = hsv_img[:, :, 1]
    val_img = hsv_img[:, :, 2]
    # create a mask then dilate it
    beige = maximum_filter(
        (
            (0.10 < sat_img) & (sat_img < 0.25)
            & (0.75 < val_img) & (val_img < 0.90)
        ).astype(np.uint8),
        disk(min(array.shape[:2]) // 150),
    )
    # find the largest region
    labels = skimage_label(beige)
    largest_region = max(
        regionprops(labels),
        key=(lambda region: region.area),
    )
    # crop
    min_row, min_col, max_row, max_col = largest_region.bbox
    return (min_row, min_col), array[min_row:max_row, min_col:max_col]


def identify_characters(array):
    # type: (NDArray) -> tuple[NDArray, dict[int, RegionProperties], dict[int, RegionProperties]]
    """Identify character and border (and artifact) regions."""
    char_regions = {}
    misc_regions = {}
    min_dimension = min(array.shape[0], array.shape[1]) // 100
    max_dimension = min(array.shape[0], array.shape[1]) // 4
    labels = skimage_label(array)
    for region in regionprops(labels):
        min_row, min_col, max_row, max_col = region.bbox
        width = max_col - min_col # the width of the region
        height = max_row - min_row # the height of the region
        if width < min_dimension and height < min_dimension:
            # discard small image artifacts
            continue
        density = region.extent
        num_holes = 1 - region.euler_number
        is_character = (
            True
            # no larger than a maximum dimension
            and width < max_dimension
            and height < max_dimension
            # aspect ratio less than 10
            and (width / height) < 10
            and (height / width) < 10
            # more than 15% of pixels are characters
            and 0.15 < density < 0.90
            # not contain too many holes
            and (num_holes / region.area) < 0.015
        )
        if is_character:
            char_regions[region.label] = region
        else:
            misc_regions[region.label] = region
    return labels, char_regions, misc_regions


def visualize(*mask_colors, background=None):
    # type: (*tuple[NDArray, Color], NDArray) -> None
    """Visualize masks on a background.

    Colors are always integer RGB tuples.

    Parameters:
        mask_colors (list[array, color]): A list of masks and their colors.
        background (array | color | None): The background image to draw on.
    """
    # create the background
    if isinstance(background, np.ndarray):
        # if it's an image, use it
        result = background.astype(np.uint8)
    else:
        if background is None:
            background = (0, 0, 0)
        # if not, create it by first figuring out the size of the masks
        height = max(mask.shape[0] for mask, _ in mask_colors)
        width = max(mask.shape[1] for mask, _ in mask_colors)
        # figure out the color as a tuple of integers
        result = (
            np.repeat(
                background,
                height * width,
            ).reshape(
                (height, width, 3),
                order='F',
            ).astype(np.uint8)
        )
    # apply each mask
    for mask, color in mask_colors:
        result = np.ma.masked_array(
            result,
            np.repeat(mask, result.shape[2]).reshape(result.shape),
            fill_value=color,
        ).filled()
    # save the image
    save_image(result, path=Path('intermediate'))
    return result


def export_all_text(array, border_mask, char_mask, crop_offset):
    # type: (NDArray, NDArray, NDArray, tuple[int, int]) -> None
    directory = Path() / 'output' / STATE['filepath'].stem
    # empty the directory
    for f in directory.glob('*.png'):
        f.unlink()
    # export each line of text
    labels = skimage_label(~border_mask)
    for region in regionprops(labels):
        min_row, min_col, max_row, max_col = region.bbox
        # ignore regions that don't have characters
        if not char_mask[min_row:max_row, min_col:max_col].any():
            continue
        row_offset, col_offset = crop_offset
        filename = '_'.join([
            STATE['filepath'].stem,
            f'r{min_row + row_offset}',
            f'c{min_col + col_offset}',
            f'r{max_row + row_offset}',
            f'c{max_col + col_offset}',
        ])
        save_image(
            array[min_row:max_row, min_col:max_col],
            path=Path() / 'output' / STATE['filepath'].stem,
            filename=(filename + '.png'),
        )


def sum_dimension(array, dimension):
    # type: (NDArray, str) -> NDArray
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


def find_basins(array, dimension, threshold):
    # type: (NDArray, str, float) -> list[tuple[int, int]]
    # count the pixels along the dimension
    pixel_counts = sum_dimension(array, dimension)
    char_ratios = pixel_counts / len(pixel_counts)
    valleys = np.nonzero(char_ratios < threshold)[0]
    basins = [] # type: list[tuple[int, int]]
    first_value = None
    prev_value = None
    # determine the start and end of basins
    for value in valleys:
        if first_value is None:
            first_value = value
        elif value > prev_value + 1:
            basins.append((first_value, prev_value))
            first_value = None
        prev_value = value
    basins.append((first_value, prev_value))
    return basins


def create_grid_node_mask(char_mask, edges):
    # type: (NDArray, list[BorderEdge]) -> NDArray
    mask = np.zeros(char_mask.shape).astype(np.uint8)
    nodes = set()
    for edge in edges:
        nodes.add(edge.node1)
        nodes.add(edge.node2)
    for node in nodes:
        mask[node.min_row:node.max_row, node.min_col:node.max_col] = 1
    return mask


def create_grid_edge_mask(char_mask, edges, style='outline'):
    # type: (NDArray, list[BorderEdge], str) -> NDArray
    assert style in ['outline', 'filled']
    mask = np.zeros(char_mask.shape).astype(np.uint8)
    if style == 'outline':
        for edge in edges:
            mask[edge.min_row, edge.min_col:edge.max_col] = 1
            mask[edge.max_row, edge.min_col:edge.max_col] = 1
            mask[edge.min_row:edge.max_row, edge.min_col] = 1
            mask[edge.min_row:edge.max_row, edge.max_col] = 1
    elif style == 'filled':
        for edge in edges:
            mask[edge.min_row:edge.max_row, edge.min_col:edge.max_col] = 1
    return mask


class Coord:

    def __init__(self, row, col):
        # type: (int, int) -> None
        assert row is not None
        assert col is not None
        self.row = row
        self.col = col

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return (
            self.row == other.row
            and self.col == other.col
        )


class BorderNode:

    def __init__(self, upper_left, lower_right):
        # type: (Coord, Coord) -> None
        self.upper_left = upper_left
        self.lower_right = lower_right
        assert self.upper_left.row <= self.lower_right.row, (self.upper_left.row, self.lower_right.row)
        assert self.upper_left.col <= self.lower_right.col, (self.upper_left.col, self.lower_right.col)

    def __hash__(self):
        return hash((self.upper_left, self.lower_right))

    def __eq__(self, other):
        return (
            self.upper_left == other.upper_left
            and self.lower_right == other.upper_left
        )

    @property
    def min_row(self):
        return self.upper_left.row

    @property
    def max_row(self):
        return self.lower_right.row

    @property
    def min_col(self):
        return self.upper_left.col

    @property
    def max_col(self):
        return self.lower_right.col

    @property
    def width(self):
        return self.lower_right.col - self.upper_left.col

    @property
    def height(self):
        return self.lower_right.row - self.upper_left.row


class BorderEdge:

    def __init__(self, node1, node2):
        # type: (BorderNode, BorderNode) -> None
        self.node1 = node1
        self.node2 = node2
        self.min_row = min(node1.min_row, node2.min_row)
        self.max_row = max(node1.max_row, node2.max_row)
        self.min_col = min(node1.min_col, node2.min_col)
        self.max_col = max(node1.max_col, node2.max_col)
        self.orientation = None
        if node1.min_row == node2.min_row and node1.max_row == node2.max_row:
            assert self.orientation is None
            self.orientation = 'horizontal'
            self.min_col += node1.width
            self.max_col -= node2.width
        if node1.min_col == node2.min_col and node1.max_col == node2.max_col:
            assert self.orientation is None
            self.orientation = 'vertical'
            self.min_row += node1.height
            self.max_row -= node2.height
        assert self.orientation is not None

    @property
    def is_horizontal(self):
        return self.orientation == 'horizontal'

    @property
    def is_vertical(self):
        return self.orientation == 'vertical'

    @property
    def width(self):
        if self.is_horizontal:
            return self.max_row - self.min_row
        else:
            return self.max_col - self.min_col

    def shrink(self, char_mask, border_mask):
        # FIXME need to be more careful here
        # the actual border - if one exists, could be to either side of the edge instead of being between them
        # need to do a broader sweep to determine the best way to adjust the border
        if self.is_horizontal:
            self.min_row = self.min_row
            min_num_chars = max(border_mask.shape)
            while self.min_row < self.max_row:
                num_border_pixels = sum(border_mask[self.min_row, self.min_col:self.max_col])
                num_char_pixels = sum(char_mask[self.min_row, self.min_col:self.max_col])
                if num_char_pixels == 0 and num_border_pixels > min_num_chars:
                    break
                min_num_chars = num_border_pixels
                self.min_row += 1
            min_num_chars = max(border_mask.shape)
            while self.max_row > self.min_row:
                num_border_pixels = sum(border_mask[self.max_row, self.min_col:self.max_col])
                num_char_pixels = sum(char_mask[self.max_row, self.min_col:self.max_col])
                if num_char_pixels == 0 and num_border_pixels > min_num_chars:
                    break
                min_num_chars = num_border_pixels
                self.max_row -= 1
        else:
            self.min_col = self.min_col
            min_num_chars = max(border_mask.shape)
            while self.min_col < self.max_col:
                num_border_pixels = sum(border_mask[self.min_row:self.max_row, self.min_col])
                num_char_pixels = sum(char_mask[self.min_row:self.max_row, self.min_col])
                if num_char_pixels == 0 and num_border_pixels > min_num_chars:
                    break
                min_num_chars = num_border_pixels
                self.min_col += 1
            min_num_chars = max(border_mask.shape)
            while self.max_col > self.min_col:
                num_border_pixels = sum(border_mask[self.min_row:self.max_row, self.max_col])
                num_char_pixels = sum(char_mask[self.min_row:self.max_row, self.max_col])
                if num_char_pixels == 0 and num_border_pixels > min_num_chars:
                    break
                min_num_chars = num_border_pixels
                self.max_col -= 1


def build_grid(char_mask, args):
    row_basins = find_basins(char_mask, 'row', args.border_ratio_threshold)
    col_basins = find_basins(char_mask, 'col', args.border_ratio_threshold)
    nodes = {} # index by upper-left coord
    for min_row, max_row in row_basins:
        for min_col, max_col in col_basins:
            upper_left = Coord(min_row, min_col)
            nodes[upper_left] = BorderNode(upper_left, Coord(max_row, max_col))
    edges = []
    prev_row = []
    for min_row, max_row in row_basins:
        curr_row = []
        prev_node = None
        for i, (min_col, max_col) in enumerate(col_basins):
            upper_left = Coord(min_row, min_col)
            node = nodes[upper_left]
            # add horizontal edge
            if prev_node is not None:
                edges.append(BorderEdge(prev_node, node))
            # add vertical edge
            if prev_row:
                edges.append(BorderEdge(prev_row[i], node))
            prev_node = node
            curr_row.append(node)
        prev_row = curr_row
    return set(nodes.values()), edges


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


def centroid_crosses_border(centroid1, centroid2, no_mans_coords):
    return any(
        (coord in no_mans_coords) for coord
        in zip(*line(*(round(x) for x in (*centroid1, *centroid2))))
    )


def k_nearest_neighbors_hash(regions, k, grid_size, no_mans_coords):
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
    all_nearest_neighbors = {} # type: dict[int, list[int]]
    for region in regions:
        key = (region.centroid[0] // grid_size, region.centroid[1] // grid_size)
        keys.append(key)
        hash_grid[key].append(region)
        all_nearest_neighbors[region.label] = []
    # initialize result variables
    distance_cache = {}
    max_radius = 3
    max_distance = max_radius * max_radius * grid_size * grid_size
    # loop over each region to look for its nearest neighbors
    for this_key, this_region in zip(keys, regions):
        this_centroid = this_region.centroid
        away_neighbors = []
        near_neighbors = []
        for radius, offsets in hash_grid_radius_offsets(max_radius):
            # pre-calculate the maximum distance we will consider, accounting for grid squareness
            radius_distance = radius * radius * grid_size * grid_size
            # loop over the grid cells in the larger radius
            for offset in offsets:
                that_key = (this_key[0] + offset[0], this_key[1] + offset[1])
                # loop over the regions in that grid cell
                for that_region in hash_grid[that_key]:
                    # skip over the original region
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
                    if distance < max_distance:
                        away_neighbors.append((distance, that_region))
            # add regions that were too far away but are now eligible
            new_away_neighbors = []
            for distance, that_region in away_neighbors:
                if distance > radius_distance:
                    new_away_neighbors.append((distance, that_region))
                    continue
                # check if connecting the centroids will cross no man's land
                if not centroid_crosses_border(this_region.centroid, that_region.centroid, no_mans_coords):
                    near_neighbors.append((distance, that_region))
            away_neighbors = new_away_neighbors
            # if there are enough near neighbors, this region is done
            if len(near_neighbors) >= k:
                break
        all_nearest_neighbors[this_region.label] = [
            that_region.label for _, that_region in sorted(near_neighbors)[:k]
        ]
        result = all_nearest_neighbors[this_region.label]
        assert len(result) == len(set(result))
    # return the list of nearest neighbors
    return all_nearest_neighbors


def k_nearest_neighbors_naive(regions, k, _, no_mans_land):
    no_mans_coords = set(zip(*np.nonzero(no_mans_land)))
    regions_dict = {region.label: region for region in regions}
    region_labels = sorted(regions_dict.keys())
    all_nearest_neighbors = {}
    for i, region_label1 in enumerate(region_labels):
        region1 = regions_dict[region_label1]
        centroid1 = region1.centroid
        neighbors = []
        for region_label2 in region_labels[i+1:]:
            region2 = regions_dict[region_label2]
            centroid2 = region2.centroid
            dx = centroid1[0] - centroid2[0]
            dy = centroid1[1] - centroid2[1]
            distance = dx * dx + dy * dy
            if not centroid_crosses_border(centroid1, centroid2, no_mans_coords):
                neighbors.append((distance, region2))
        all_nearest_neighbors[region_label1] = [
            region2.label for _, region2 in sorted(neighbors)[:k]
        ]
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
    # initialize the image to all black
    array = np.zeros((*labels.shape, 3)).astype(np.uint8)
    # for each component
    for component in components:
        # randomly pick a color for this component
        rgb = (
            RNG.randrange(128, 255),
            RNG.randrange(128, 255),
            RNG.randrange(128, 255),
        )
        # color the regions
        region_mask = np.isin(
            labels,
            [regions[region_index].label for region_index in component],
        )
        array = np.ma.masked_array(
            array,
            np.repeat(region_mask, array.shape[2]).reshape(array.shape),
            fill_value=rgb,
        ).filled()
        # color the bounding box
        min_rows, min_cols, max_rows, max_cols = zip(*(
            regions[region_index].bbox for region_index in component
        ))
        perimeter_mask = rectangle_perimeter(
            start=(min(min_rows), min(min_cols)),
            end=(max(max_rows), max(max_cols)),
            shape=array.shape,
            clip=True,
        )
        array[perimeter_mask] = rgb
    save_image(array)
    return array


def get_ocr_results(image_path, crop_offset):
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir).resolve()
        run(
            [
                str(Path('run-ocr.sh').resolve()),
                '--sourceimg', str(image_path),
                '--output', str(temp_dir_path),
            ],
            check=True,
        )
        json_path = temp_dir_path / image_path.with_suffix('.json').name
        with json_path.open() as fd:
            json = open_json(fd)
    bboxes = []
    for text_area in json['contents'][0]:
        min_col, min_row = text_area['boundingBox'][0]
        max_col, max_row = text_area['boundingBox'][-1]
        min_row -= crop_offset[0]
        min_col -= crop_offset[1]
        bboxes.append((text_area['id'], (min_col, min_row, max_col, max_row)))
    return bboxes


def pipeline(path, args):
    # type: (Path, dict[str, Any]) -> None
    reset_state()
    STATE['filepath'] = path
    # read the image
    array = imread(path)
    visualize(background=array)
    check_time('read in the image')
    # crop to just the page
    if args.crop:
        crop_offset, cropped_image = crop(array)
    else:
        crop_offset = (0, 0)
        cropped_image = array
    visualize(background=cropped_image)
    check_time('cropped the image')
    # convert to black-and-white
    array = cropped_image
    array = (rgb2gray(array) * 255 > 127) * np.ones(array.shape[:2])
    array = (array * 255).astype(np.uint8)
    # separate characters from borders
    array = invert(array)
    labels, char_regions, misc_regions = identify_characters(array)
    char_mask = np.isin(labels, list(char_regions.keys()))
    misc_mask = np.isin(labels, list(misc_regions.keys()))
    visualize(
        (misc_mask, (255, 255, 255)),
        (char_mask, (0, 255, 0)),
    )
    check_time('visualized characters and borders')
    # get OCR text regions
    ocr_bboxes = get_ocr_results(path, crop_offset)
    # associate characters with OCR regions (FIXME unoptimized)
    char_to_ocr = defaultdict(set)
    ocr_to_char = defaultdict(set)
    ocr_mask = np.zeros(char_mask.shape).astype(np.uint8)
    for ocr_id, (ocr_min_col, ocr_min_row, ocr_max_col, ocr_max_row) in ocr_bboxes:
        if ocr_max_col == ocr_mask.shape[1]:
            ocr_max_col -= 1
        if ocr_max_row == ocr_mask.shape[0]:
            ocr_max_row -= 1
        ocr_mask[ocr_min_row, ocr_min_col:ocr_max_col] = 1
        ocr_mask[ocr_max_row, ocr_min_col:ocr_max_col] = 1
        ocr_mask[ocr_min_row:ocr_max_row, ocr_min_col] = 1
        ocr_mask[ocr_min_row:ocr_max_row, ocr_max_col] = 1
        for region_id, region in char_regions.items():
            char_min_row, char_min_col, char_max_row, char_max_col = region.bbox
            intersects = (
                (
                    char_min_row <= ocr_min_row < char_max_row
                    or ocr_min_row <= char_min_row < ocr_max_row
                ) and (
                    char_min_col <= ocr_min_col < char_max_col
                    or ocr_min_col <= char_min_col < ocr_max_col
                )
            )
            if intersects:
                char_to_ocr[region_id].add(ocr_id)
                ocr_to_char[ocr_id].add(region_id)
    visualize(
        (misc_mask, (255, 255, 255)),
        (char_mask, (0, 255, 0)),
        (ocr_mask, (0, 0, 255)),
    )
    # find rows and columns where there are no characters
    # the character mask has a 1 where there are characters and 0 where there aren't
    nodes, edges = build_grid(char_mask, args)
    visualize(
        (misc_mask, (255, 255, 255)),
        (char_mask, (0, 255, 0)),
        (
            create_grid_node_mask(char_mask, edges),
            (0, 0, 255),
        ),
        (
            create_grid_edge_mask(char_mask, edges),
            (255, 0, 0),
        ),
    )
    check_time('visualized grid')
    # shrink the edges to segment attached characters
    for edge in edges:
        edge.shrink(char_mask, misc_mask)
    edge_mask = create_grid_edge_mask(char_mask, edges, style='filled')
    for node in nodes:
        edge_mask[node.min_row:node.max_row, node.min_col:node.max_col] = 1
    visualize(
        (misc_mask, (255, 255, 255)),
        (char_mask, (0, 255, 0)),
        (edge_mask, (255, 0, 0)),
    )
    check_time('visualized edge-minimized grid')
    # mark non-edge border regions as characters
    labels, char_regions, misc_regions = identify_characters(
        char_mask | (misc_mask & ~edge_mask).astype(bool)
    )
    char_mask = np.isin(labels, list(char_regions.keys()))
    misc_mask = (misc_mask & ~char_mask)
    visualize(
        (misc_mask, (255, 255, 255)),
        (char_mask, (0, 255, 0)),
    )
    check_time('updated characters')
    # mark small connected regions as their respective larger regions
    # FIXME
    # crop out individual character lines
    export_all_text(
        cropped_image,
        (
            create_grid_node_mask(char_mask, edges)
            | create_grid_edge_mask(char_mask, edges, style='filled')
        ),
        char_mask,
        crop_offset,
    )

    return # FIXME
    # find nearest neighbors and visualize
    misc_mask = np.zeros(labels.shape).astype(bool)
    misc_mask[np.isin(labels, list(misc_regions.keys()))] = True
    border_coords = set(zip(*np.nonzero(misc_mask)))
    nearest_neighbors = k_nearest_neighbors_hash(
        char_regions.values(),
        args.k,
        min(array.shape[0], array.shape[1]) // 20,
        border_coords,
    )
    check_time('found the k nearest neighbors')
    components = find_connected_components(nearest_neighbors)
    check_time('found the connected components')
    visualize_components(char_regions, labels, components)
    check_time('visualized connected components')


def main():
    # type: () -> None
    arg_parser = ArgumentParser()
    arg_parser.add_argument('images', metavar='image', type=Path, nargs='+')
    arg_parser.add_argument('-k', default=3, type=int)
    arg_parser.add_argument('--crop', action='store_true')
    arg_parser.add_argument('--border-ratio-threshold', default=0.05, type=float)
    args = arg_parser.parse_args()
    args.images = sorted(set(path.expanduser().resolve() for path in args.images))
    for image_path in args.images:
        print(image_path)
        pipeline(image_path, args)
        print()


if __name__ == '__main__':
    main()
