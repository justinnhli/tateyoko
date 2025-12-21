#!/usr/bin/env python3
"""
The first pass of this program aims to find intersections where gap rows/cols meet gap rows/cols.
The second pass then validates that gaps continue between consecutive intersections.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import signal


@dataclass
class Intersection:
    """Represents a grid intersection point where a gap row meets a gap column."""
    # the row and column index of this intersection point
    row: int
    col: int
    # ratio of white pixels in the row (0.0 = all black, 1.0 = all white)
    row_white_ratio: float
    # ratio of white pixels in the column (0.0 = all black, 1.0 = all white)
    col_white_ratio: float
    # whether this intersection passed validation checks
    is_valid: bool
    # how continous/strong the gap is at this intersection (1 - row_white_ratio)
    gap_strength: float
    
    # boolean flags indicating if the gap continues in each direction from this point
    gap_continues_left: bool = False
    gap_continues_right: bool = False
    gap_continues_up: bool = False
    gap_continues_down: bool = False
    
    # the column/row indices of neighboring intersections in each direction
    left_col: int = None
    right_col: int = None
    above_row: int = None
    below_row: int = None
    
    # how strong the gap is when traveling toward each neighbor (0.0 to 1.0)
    left_gap_ratio: float = 0.0
    right_gap_ratio: float = 0.0
    up_gap_ratio: float = 0.0
    down_gap_ratio: float = 0.0


class IntersectionAnalyzer:
    """Identifies grid intersections where gaps meet gaps in a binary image."""
    
    def __init__(self, binary_array, window_size=10, min_distance=10, 
                 detection_threshold=0.3, intersection_threshold=0.3, 
                 gap_continuation_threshold=0.7, text_threshold=0.5):
        """Initialize the analyzer with a binary image and parameters.
        
        args:
            binary_array: binary array where 1=white/text, 0=black/background
            window_size: size of validation window around each intersection point
            min_distance: minimum pixel distance between detected peaks/valleys
            detection_threshold: max white ratio to consider as a gap row/column
            intersection_threshold: max white ratio to consider as gap at intersection
            gap_continuation_threshold: min black ratio to consider gap as continuing
            text_threshold: max white ratio to consider a gap segment as having "no text"
        """
        self.binary = binary_array
        self.window_size = window_size
        self.min_distance = min_distance
        self.detection_threshold = detection_threshold
        self.intersection_threshold = intersection_threshold
        self.gap_continuation_threshold = gap_continuation_threshold
        self.text_threshold = text_threshold
        self.intersections = []
    
    def sum_dimension(self, array, dimension):
        """Sums all white pixels along each row or column.
            - returns an array where each element is the total white pixels in that row/col.
        """
        if dimension == 'row':
            return np.sum(array, axis=1)
        elif dimension == 'col':
            return np.sum(array, axis=0)
        else:
            raise ValueError(f'dimension should be "row" or "col", instead got "{dimension}"')
    
    def find_peaks_and_valleys(self, signal_data, threshold, min_distance=10) -> Tuple[List[int], List[int]]:
        """Find peaks (high white pixel counts) and valleys (low white pixel counts).
            - peaks are rows/columns with lots of white (text), valleys are mostly black (gaps).
            merges consecutive peaks/valleys together and returns their center indices.
        """
        peaks = []
        valleys = []
        
        # first pass: classify each index as peak or valley based on threshold
        for i in range(len(signal_data)):
            if signal_data[i] >= threshold:
                peaks.append(i)
            else:
                valleys.append(i)
        
        def merge_consecutive(indices, min_dist):
            """Merge clusters of consecutive indices and return their centers."""
            if not indices:
                return []
            merged = []
            current_group = [indices[0]]
            
            # group together indices that are close to each other
            for idx in indices[1:]:
                if idx - current_group[-1] <= min_dist:
                    current_group.append(idx)
                else:
                    # when we find a gap, save the center of the group and start a new group
                    # this is the midpoint method 
                    merged.append(current_group[len(current_group) // 2])
                    current_group = [idx]
            
            # to include the last group
            if current_group:
                merged.append(current_group[len(current_group) // 2])
            
            return merged
        
        peaks = merge_consecutive(peaks, min_distance)
        valleys = merge_consecutive(valleys, min_distance)
        
        return peaks, valleys
    
    def analyze(self) -> Tuple[List[int], List[int], List[int], List[int], List[Intersection], List[Intersection]]:
        """Execute the complete analysis pipeline from start to finish."""
        # count white pixels in each row and column
        row_white_counts = self.sum_dimension(self.binary, 'row')
        col_white_counts = self.sum_dimension(self.binary, 'col')
        
        # convert counts to ratios (0.0 to 1.0)
        row_white_ratios = row_white_counts / self.binary.shape[1]
        col_white_ratios = col_white_counts / self.binary.shape[0]
        
        # print additional statistics about the image for debugging 
        print(f"row white ratios: min={row_white_ratios.min():.3f}, max={row_white_ratios.max():.3f}, mean={row_white_ratios.mean():.3f}")
        print(f"col white ratios: min={col_white_ratios.min():.3f}, max={col_white_ratios.max():.3f}, mean={col_white_ratios.mean():.3f}")
        
        # find peaks (text rows) and valleys (gap rows)
        print("detecting peaks and valleys in rows...")
        peak_rows, valley_rows = self.find_peaks_and_valleys(
            row_white_ratios, 
            self.detection_threshold, 
            self.min_distance
        )
        
        # find peaks (text columns) and valleys (gap columns)
        print("detecting peaks and valleys in columns...")
        peak_cols, valley_cols = self.find_peaks_and_valleys(
            col_white_ratios, 
            self.detection_threshold, 
            self.min_distance
        )

        # print additional statistics about the image for debugging 
        print(f"found {len(peak_rows)} peak rows, {len(valley_rows)} valley rows")
        print(f"found {len(peak_cols)} peak cols, {len(valley_cols)} valley cols")
        
        # find all intersections and filter to valid ones
        all_intersections, valid_intersections = self.run(peak_rows, valley_rows, peak_cols, valley_cols)
        
        return peak_rows, valley_rows, peak_cols, valley_cols, all_intersections, valid_intersections
    
    def run(self, peak_rows, valley_rows, peak_cols, valley_cols):
        """
        second pass: analyze intersections between gap rows and gap columns.
        
        1. creates intersection points at all combinations of peaks/valleys,
        2. then validates that gaps actually continue between adjacent intersections.
        """
        intersections = []
        
        row_white_counts = self.sum_dimension(self.binary, 'row')
        col_white_counts = self.sum_dimension(self.binary, 'col')
        
        row_white_ratios = row_white_counts / self.binary.shape[1]
        col_white_ratios = col_white_counts / self.binary.shape[0]
        
        print(f"debug: creating intersections at all peak/valley combinations")
        
        # create intersections at all combinations of (peak row, peak col)
        for row in peak_rows:
            for col in peak_cols:
                if row < 0 or row >= len(row_white_ratios) or col < 0 or col >= len(col_white_ratios):
                    continue
                
                row_white_ratio = row_white_ratios[row]
                col_white_ratio = col_white_ratios[col]
                gap_strength = 1.0 - row_white_ratio
                
                intersection = Intersection(
                    row=row, col=col, row_white_ratio=row_white_ratio,
                    col_white_ratio=col_white_ratio, is_valid=True, gap_strength=gap_strength,
                )
                intersections.append(intersection)
        
        # create intersections at all combinations of (valley row, peak col)
        for row in valley_rows:
            for col in peak_cols:
                if row < 0 or row >= len(row_white_ratios) or col < 0 or col >= len(col_white_ratios):
                    continue
                
                row_white_ratio = row_white_ratios[row]
                col_white_ratio = col_white_ratios[col]
                gap_strength = 1.0 - row_white_ratio
                
                intersection = Intersection(
                    row=row, col=col, row_white_ratio=row_white_ratio,
                    col_white_ratio=col_white_ratio, is_valid=True, gap_strength=gap_strength,
                )
                intersections.append(intersection)
        
        # create intersections at all combinations of (peak row, valley col)
        for row in peak_rows:
            for col in valley_cols:
                if row < 0 or row >= len(row_white_ratios) or col < 0 or col >= len(col_white_ratios):
                    continue
                
                row_white_ratio = row_white_ratios[row]
                col_white_ratio = col_white_ratios[col]
                gap_strength = 1.0 - row_white_ratio
                
                intersection = Intersection(
                    row=row, col=col, row_white_ratio=row_white_ratio,
                    col_white_ratio=col_white_ratio, is_valid=True, gap_strength=gap_strength,
                )
                intersections.append(intersection)
        
        # create intersections at all combinations of (valley row, valley col)
        for row in valley_rows:
            for col in valley_cols:
                if row < 0 or row >= len(row_white_ratios) or col < 0 or col >= len(col_white_ratios):
                    continue
                
                row_white_ratio = row_white_ratios[row]
                col_white_ratio = col_white_ratios[col]
                gap_strength = 1.0 - row_white_ratio
                
                intersection = Intersection(
                    row=row, col=col, row_white_ratio=row_white_ratio,
                    col_white_ratio=col_white_ratio, is_valid=True, gap_strength=gap_strength,
                )
                intersections.append(intersection)
        
        print(f"debug: total intersections before validation: {len(intersections)}")
        
        # validate that gaps actually continue between adjacent intersections
        for intersection in intersections:
            row = intersection.row
            col = intersection.col
            
            # find other intersections in the same row and column
            intersections_in_same_row = [
                inter for inter in intersections 
                if inter.row == row and inter.col != col
            ]
            
            intersections_in_same_col = [
                inter for inter in intersections 
                if inter.col == col and inter.row != row
            ]
            
            # check if this row/column is a gap or a text row/column
            is_gap_row = row in valley_rows
            is_gap_col = col in valley_cols
            
            # validate horizontal gap continuations (left and right)
            if intersections_in_same_row:
                # check for a neighbor to the left
                left_neighbors = [inter for inter in intersections_in_same_row if inter.col < col]
                if left_neighbors:
                    left_neighbor = max(left_neighbors, key=lambda x: x.col)
                    intersection.left_col = left_neighbor.col
                    
                    # if this is a gap row, check if the segment between neighbors is mostly black
                    if is_gap_row:
                        row_segment = self.binary[row, left_neighbor.col:col]
                        if len(row_segment) > 0:
                            white_pixels = np.sum(row_segment == 1)
                            white_ratio = white_pixels / len(row_segment)
                            intersection.gap_continues_left = white_ratio <= self.text_threshold
                            intersection.left_gap_ratio = 1.0 - white_ratio
                    else:
                        # if this is a text row, we don't need to check the segment
                        intersection.gap_continues_left = True
                else:
                    # no left neighbor - auto-validate if this is a gap row
                    if is_gap_row:
                        intersection.gap_continues_left = True
                
                # check for a neighbor to the right
                right_neighbors = [inter for inter in intersections_in_same_row if inter.col > col]
                if right_neighbors:
                    right_neighbor = min(right_neighbors, key=lambda x: x.col)
                    intersection.right_col = right_neighbor.col
                    
                    # if this is a gap row, check if the segment between neighbors is mostly black
                    if is_gap_row:
                        row_segment = self.binary[row, col:right_neighbor.col]
                        if len(row_segment) > 0:
                            white_pixels = np.sum(row_segment == 1)
                            white_ratio = white_pixels / len(row_segment)
                            intersection.gap_continues_right = white_ratio <= self.text_threshold
                            intersection.right_gap_ratio = 1.0 - white_ratio
                    else:
                        # if this is a text row, we don't need to check the segment
                        intersection.gap_continues_right = True
                else:
                    # no right neighbor so auto-validate if this is a gap row
                    if is_gap_row:
                        intersection.gap_continues_right = True
            else:
                # no neighbors in same row so auto-validate if this is a gap row
                if is_gap_row:
                    intersection.gap_continues_left = True
                    intersection.gap_continues_right = True
            
            # validate vertical gap continuations (up and down)
            if intersections_in_same_col:
                # check for a neighbor above
                up_neighbors = [inter for inter in intersections_in_same_col if inter.row < row]
                if up_neighbors:
                    up_neighbor = max(up_neighbors, key=lambda x: x.row)
                    intersection.above_row = up_neighbor.row
                    
                    # if this is a gap column, check if the segment between neighbors is mostly black
                    if is_gap_col:
                        col_segment = self.binary[up_neighbor.row:row, col]
                        if len(col_segment) > 0:
                            white_pixels = np.sum(col_segment == 1)
                            white_ratio = white_pixels / len(col_segment)
                            intersection.gap_continues_up = white_ratio <= self.text_threshold
                            intersection.up_gap_ratio = 1.0 - white_ratio
                    else:
                        # if this is a text column, we don't need to check the segment
                        intersection.gap_continues_up = True
                else:
                    # no upper neighbor so auto-validate if this is a gap column
                    if is_gap_col:
                        intersection.gap_continues_up = True
                
                # check for a neighbor below
                down_neighbors = [inter for inter in intersections_in_same_col if inter.row > row]
                if down_neighbors:
                    down_neighbor = min(down_neighbors, key=lambda x: x.row)
                    intersection.below_row = down_neighbor.row
                    
                    # if this is a gap column, check if the segment between neighbors is mostly black
                    if is_gap_col:
                        col_segment = self.binary[row:down_neighbor.row, col]
                        if len(col_segment) > 0:
                            white_pixels = np.sum(col_segment == 1)
                            white_ratio = white_pixels / len(col_segment)
                            intersection.gap_continues_down = white_ratio <= self.text_threshold
                            intersection.down_gap_ratio = 1.0 - white_ratio
                    else:
                        # if this is a text column, we don't need to check the segment
                        intersection.gap_continues_down = True
                else:
                    # no lower neighbor so auto-validate if this is a gap column
                    if is_gap_col:
                        intersection.gap_continues_down = True
            else:
                # no neighbors in same column so auto-validate if this is a gap column
                if is_gap_col:
                    intersection.gap_continues_up = True
                    intersection.gap_continues_down = True
        
        # filter intersections to keep only those with at least one valid gap continuation
        valid_intersections = []
        for intersection in intersections:
            has_horizontal_continuation = intersection.gap_continues_left or intersection.gap_continues_right
            has_vertical_continuation = intersection.gap_continues_up or intersection.gap_continues_down
            
            if has_horizontal_continuation or has_vertical_continuation:
                valid_intersections.append(intersection)
        
        print(f"debug: valid intersections after gap continuation check: {len(valid_intersections)}/{len(intersections)}")
        
        self.intersections = valid_intersections
        return intersections, valid_intersections


def extract_subvalleys_at_intersections(binary_array, gap_rows, gap_cols, window_size=20):
    """Extract sub-valleys (black regions) around each intersection point.
        - for each intersection, takes a square window and counts black pixels.
    """
    subvalleys = []
    all_coordinates = []
    
    # for each intersection point, analyze the region around it
    for row in gap_rows:
        for col in gap_cols:
            # define a square window around the intersection
            row_start = max(0, row - window_size)
            row_end = min(binary_array.shape[0], row + window_size)
            col_start = max(0, col - window_size)
            col_end = min(binary_array.shape[1], col + window_size)
            
            # extract the window from the image
            window = binary_array[row_start:row_end, col_start:col_end]
            
            # count black pixels in the window
            black_pixels = np.sum(window == 0)
            total_pixels = window.size
            
            if total_pixels > 0:
                black_ratio = black_pixels / total_pixels
                
                # store information about this sub-valley
                subvalley = {
                    'row': row,
                    'col': col,
                    'window_bounds': (row_start, row_end, col_start, col_end),
                    'black_pixels': black_pixels,
                    'total_pixels': total_pixels,
                    'black_ratio': black_ratio,
                }
                subvalleys.append(subvalley)
                
                # also collect the actual coordinates of all black pixels
                rows, cols = np.where(window == 0)
                for r, c in zip(rows, cols):
                    all_coordinates.append((row_start + r, col_start + c))
    
    return {
        'subvalleys': subvalleys,
        'all_coordinates': all_coordinates,
    }


def report_subvalleys(subvalleys):
    """Print a formatted report of all detected sub-valleys."""
    if not subvalleys:
        print("no sub-valleys detected.")
        return
    
    print(f"\ndetected {len(subvalleys)} sub-valleys:")
    
    # print details for each sub-valley
    for i, sv in enumerate(subvalleys):
        print(f"sub-valley {i+1}:")
        print(f"  position: row={sv['row']}, col={sv['col']}")
        print(f"  black ratio: {sv['black_ratio']:.3f}")
        print(f"  black pixels: {sv['black_pixels']}/{sv['total_pixels']}")
    
