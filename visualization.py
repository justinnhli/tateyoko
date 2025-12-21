#!/usr/bin/env python3
"""Visualization script for intersection analyzer."""

import sys
import numpy as np
import cv2
from subvalley import IntersectionAnalyzer
from skimage.util import invert
from skimage.measure import label as skimage_label, regionprops
from skimage.morphology import flood

def load_image(image_path):
    """Load an image and convert it to a binary array (0 and 1 only)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"could not load image: {image_path}")
    
    # convert grayscale image to binary using threshold at 128
    # pixels < 128 become 0 (black), pixels >= 128 become 1 (white)
    _, binary = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)
    return binary, img

def get_crop_bounds(array):
    """Find the bounding box of the main document content, ignoring white borders."""
    # flood fill from the top-left corner to find all connected white background
    flood_mask = flood(array, (0, 0))
    # invert to get the document content (not the background)
    flooded = (invert(flood_mask) * np.ones(array.shape) * 255).astype(np.uint8)
    # label connected regions
    labels = skimage_label(flooded)
    
    # if no regions found, return the entire image
    if len(regionprops(labels)) == 0:
        return 0, array.shape[0], 0, array.shape[1]
    
    # find the largest region (the main document content)
    largest_region = max(regionprops(labels), key=lambda region: region.area)
    # get the bounding box of this region
    min_row, min_col, max_row, max_col = largest_region.bbox
    return min_row, max_row, min_col, max_col

def visualize_results(original_image, analyzer_results, output_path="output_visualization.png"):
    """Draws the analysis results on the original cropped image and saves it.
        - draws: detected rows/cols as blue lines, gap continuations as cyan lines, all intersections as gray dots, 
        valid intersections as green dots.
    """
    # create a color version of the original grayscale image
    img_color = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    peak_rows, valley_rows, peak_cols, valley_cols, intersections, valid_intersections = analyzer_results
    
    # draw all detected rows (both peaks and valleys) as blue horizontal lines
    for row in sorted(set(list(peak_rows) + list(valley_rows))):
        row = max(0, min(row, img_color.shape[0] - 1))
        cv2.line(img_color, (0, row), (img_color.shape[1], row), color=(255, 0, 0), thickness=2)
    
    # draw all detected columns (both peaks and valleys) as blue vertical lines
    for col in sorted(set(list(peak_cols) + list(valley_cols))):
        col = max(0, min(col, img_color.shape[1] - 1))
        cv2.line(img_color, (col, 0), (col, img_color.shape[0]), color=(255, 0, 0), thickness=2)
    
    # draw gap continuations as cyan lines connecting adjacent intersections
    for intersection in intersections:
        # get the pixel coordinates of the intersection
        x, y = max(0, min(intersection.col, img_color.shape[1] - 1)), max(0, min(intersection.row, img_color.shape[0] - 1))
        
        # draw a line to the left neighbor if gap continues leftward
        if intersection.gap_continues_left and intersection.left_col is not None:
            left_col = max(0, min(intersection.left_col, img_color.shape[1] - 1))
            cv2.line(img_color, (left_col, y), (x, y), color=(255, 255, 0), thickness=4)
        
        # draw a line to the right neighbor if gap continues rightward
        if intersection.gap_continues_right and intersection.right_col is not None:
            right_col = max(0, min(intersection.right_col, img_color.shape[1] - 1))
            cv2.line(img_color, (x, y), (right_col, y), color=(255, 255, 0), thickness=4)
        
        # draw a line upward if gap continues upward
        if intersection.gap_continues_up and intersection.above_row is not None:
            above_row = max(0, min(intersection.above_row, img_color.shape[0] - 1))
            cv2.line(img_color, (x, above_row), (x, y), color=(255, 255, 0), thickness=4)
        
        # draw a line downward if gap continues downward
        if intersection.gap_continues_down and intersection.below_row is not None:
            below_row = max(0, min(intersection.below_row, img_color.shape[0] - 1))
            cv2.line(img_color, (x, y), (x, below_row), color=(255, 255, 0), thickness=4)
    
    # draw all intersections (including invalid ones) as small gray dots
    for intersection in intersections:
        x, y = max(0, min(intersection.col, img_color.shape[1] - 1)), max(0, min(intersection.row, img_color.shape[0] - 1))
        cv2.circle(img_color, (x, y), radius=3, color=(100, 100, 100), thickness=-1)
    
    # draw valid intersections as larger green dots (on top of gray dots)
    for intersection in valid_intersections:
        x, y = max(0, min(intersection.col, img_color.shape[1] - 1)), max(0, min(intersection.row, img_color.shape[0] - 1))
        cv2.circle(img_color, (x, y), radius=6, color=(0, 255, 0), thickness=-1)
    
    # save the visualization to a file
    cv2.imwrite(output_path, img_color)
    print(f"visualization saved to: {output_path}")

def print_statistics(analyzer_results, total_intersections_before):
    """print a summary of the analysis results."""
    peak_rows, valley_rows, peak_cols, valley_cols, valid_intersections = analyzer_results
    
    print("intersection analyzer results")
    print(f"peak rows (text rows): {len(peak_rows)}")
    print(f"valley rows (gap rows): {len(valley_rows)}")
    print(f"peak columns (text columns): {len(peak_cols)}")
    print(f"valley columns (gap columns): {len(valley_cols)}")
    print(f"\ntotal intersections (before): {total_intersections_before}")
    print(f"valid intersections (after): {len(valid_intersections)}")
    print(f"filtered out: {total_intersections_before - len(valid_intersections)}")
    
    # show what percentage of intersections passed validation
    if total_intersections_before > 0:
        kept_pct = (len(valid_intersections) / total_intersections_before) * 100
        print(f"kept percentage: {kept_pct:.1f}%")

def main():
    """Main entry point for the visualization script."""
    # print input information if no arguments provided
    if len(sys.argv) < 2:
        print("please input: python visualization.py <image_path> [output_path] [options]")
        print("\nexample:")
        print("  python visualization.py input.png")
        print("  python visualization.py input.png output.png --detection-threshold 0.2")
        print("\noptions:")
        print("  --detection-threshold FLOAT: max white ratio for row/col detection (default: 0.3)")
        print("  --min-distance INT: minimum distance between peaks/valleys (default: 10)")
        print("  --text-threshold FLOAT: max white ratio for gap segments (default: 0.5)")
        sys.exit(1)
    
    # parse command line arguments
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else "output_visualization.png"
    
    # default parameter values
    detection_threshold = 0.3
    min_distance = 10
    text_threshold = 0.5
    
    # parse optional parameters
    i = 2 if output_path != "output_visualization.png" else 2
    while i < len(sys.argv):
        if sys.argv[i] == '--detection-threshold' and i + 1 < len(sys.argv):
            detection_threshold = float(sys.argv[i + 1])
            print(f"set detection_threshold to {detection_threshold}")
            i += 2
        elif sys.argv[i] == '--min-distance' and i + 1 < len(sys.argv):
            min_distance = int(sys.argv[i + 1])
            print(f"set min_distance to {min_distance}")
            i += 2
        elif sys.argv[i] == '--text-threshold' and i + 1 < len(sys.argv):
            text_threshold = float(sys.argv[i + 1])
            print(f"set text_threshold to {text_threshold}")
            i += 2
        else:
            i += 1
    
    # load the image and convert to binary
    print(f"loading image: {image_path}")
    binary_array, original_image = load_image(image_path)
    print(f"image shape: {binary_array.shape}")
    
    # find the document boundaries (ignoring white borders)
    print("calculating crop bounds...")
    crop_bounds = get_crop_bounds(binary_array)
    min_row, max_row, min_col, max_col = crop_bounds
    print(f"crop bounds: rows [{min_row}:{max_row}], cols [{min_col}:{max_col}]")
    
    # create and initialize the analyzer
    print("initializing intersection analyzer...")
    print(f"parameters: detection_threshold={detection_threshold}, min_distance={min_distance}, text_threshold={text_threshold}")
    analyzer = IntersectionAnalyzer(
        binary_array,
        min_distance=min_distance,
        detection_threshold=detection_threshold,
        text_threshold=text_threshold
    )
    
    # run the analysis
    print("analyzing image...")
    try:
        peak_rows, valley_rows, peak_cols, valley_cols, all_intersections, valid_intersections = analyzer.analyze()
        print(f"analysis returned: {len(peak_rows)} peak rows, {len(valley_rows)} valley rows")
        print(f"analysis returned: {len(peak_cols)} peak cols, {len(valley_cols)} valley cols")
        print(f"analysis returned: {len(all_intersections)} intersections")
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # add the image borders as additional gap rows/columns and reanalyze
    print("adding border gaps and re-analyzing intersections...")
    peak_rows = list(peak_rows)
    valley_rows = list(valley_rows)
    peak_cols = list(peak_cols)
    valley_cols = list(valley_cols)
    
    # add the image edges and document boundaries as gaps
    valley_rows.extend([0, binary_array.shape[0] - 1, min_row, max_row])
    valley_cols.extend([0, binary_array.shape[1] - 1, min_col, max_col])
    
    # remove duplicates and sort
    peak_rows = sorted(set(peak_rows))
    valley_rows = sorted(set(valley_rows))
    peak_cols = sorted(set(peak_cols))
    valley_cols = sorted(set(valley_cols))
    
    print(f"after adding borders: {len(peak_rows)} peak rows, {len(valley_rows)} valley rows")
    print(f"after adding borders: {len(peak_cols)} peak cols, {len(valley_cols)} valley cols")
    
    # re-run analysis with the updated rows/columns
    all_intersections, valid_intersections = analyzer.run(peak_rows, valley_rows, peak_cols, valley_cols)
    print(f"after re-analyzing with borders: {len(all_intersections)} total intersections, {len(valid_intersections)} valid")
    
    # create the visualization and save it
    print("generating visualization...")
    analyzer_results = (peak_rows, valley_rows, peak_cols, valley_cols, all_intersections, valid_intersections)
    visualize_results(original_image, analyzer_results, output_path)
    
    # print stats
    print_statistics((peak_rows, valley_rows, peak_cols, valley_cols, valid_intersections), len(all_intersections))
    
    print("analysis complete!")

if __name__ == "__main__":
    main()