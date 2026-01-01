This document tries to illustrate my thought process as I work through student code for a research
project. Since I am familiar with the broader goals of the project (and know roughly what the code
is trying to do), my approach is to work bottom up. I first do the mechanical tasks of simplifying
and refactoring the code, which also familiarizes me with how the code works and makes the code more
understandable. These more software-engineering-y changes can be found in the commits leading up to
this one, and I will can talk through them in a synchronous meeting. 

Instead, this document is the second part of the process, where I sit back and consider the big
picture of whether the code is approaching the problem in the right way, or if there is a more
efficient/better way to get the right results. Just to summarize, the goal of the overall project is
to improvement digitization accuracy for some 18th century Japanese texts, specifically to identify
lines of text which can then be cropped out for OCR. The approach is to first use heuristics to
identify characters, then in the current step, identify the gaps between characters. A working
assumption is that there will be bigger and more consistent gaps between lines of text than between
characters in the same line; these gaps can then be used to refine what we consider characters, and
the process can iterate until both characters and boundaries are stable.

To identify boundaries, we are looking at the proportion of character pixels in each row/column. The
rows/columns where this number is low - the valleys - are then candidates for boundaries. Much of
the work of identifying these valleys (or really, inverse "peaks") is done by
`scipy.signal.find_peaks`, which has documentation at:

<https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>

We call the function with:

```python
find_peaks(
    -pixel_counts,
    distance=min(array.shape[0], array.shape[1]) // 50,
)
```

The documentation says that `find_peaks()` simply finds peaks by looking at whether `signal[i]` is
greater than both `signal[i-1]` and `signal[i+1]`. One implication of this algorithm is that the
peaks are necessarily solitary: it would not count as a peak if `signal[i-1] == signal[i] ==
signal[i+1]` (although there is a `plateau_size` parameter that could deal with this). Additionally,
the `distance` parameter requires the results to be separated by that value, which works by removing
shorter peaks first. This result is further filtered to remove peaks that have more than 30% of the
row/column be characters, although empirically this does not drastically change the number of peaks.
The visualization corroborates this understanding: the highlighted rows and columns are all only a
single-pixel wide, separated by some distance. 

We can think about the same algorithm graphically by looking at the plot of pixel ratios, keeping in
mind that because we negate the pixel counts, the "peaks" are actually valleys. So `find_peaks()`
would return the index of the deepest part of each valley (abiding by the `distance` constraint),
and we then only return the valleys that are deeper than 0.3. This visual understanding of the
algorithm makes obvious several places where it's weird:

* The y-axis suggests that the vast majority of rows/columns have less than 30% character pixels, so
the filter is not doing much. The bottom of most valleys have less than 5% character pixels,
although ideally we may want to dynamically determine what this threshold should be.

* Keeping in mind that the next step is to broaden the boundaries

- presumably to the same 30%
threshold - this similarly suggests the threshold is too high. Unfortunately, the plot also shows
that there is no
