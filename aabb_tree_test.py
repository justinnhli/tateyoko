from hypothesis import strategies as strats, given

from aabb_tree import BoundingBox, AABBTree


def test_bbox_overlapping_middle():
    # type: () -> None
    """Test the intersection of two bounding boxes that only overlap in the middle."""
    bbox1 = BoundingBox(-3, -1, 3, 1)
    bbox2 = BoundingBox(-1, -3, 1, 3)
    assert bbox1.area == bbox2.area
    assert (
        bbox1.intersects(bbox2)
        == bbox2.intersects(bbox1)
        == True
    )
    assert (
        bbox1.intersection(bbox2)
        == bbox2.intersection(bbox1)
        == BoundingBox(-1, -1, 1, 1)
    )


@strats.composite
def bounding_boxes(draw):
    # type: (strats.DrawFn) -> BoundingBox
    x_range = sorted(draw(strats.tuples(
        strats.integers(-100, 100),
        strats.integers(-100, 100),
    )))
    y_range = sorted(draw(strats.tuples(
        strats.integers(-100, 100),
        strats.integers(-100, 100),
    )))
    return BoundingBox(x_range[0], y_range[0], x_range[1], y_range[1])


@given(strats.lists(bounding_boxes()))
def test_aabb_tree_add(bboxes: list[BoundingBox]) -> None:
    answer = []
    for i, bbox1 in enumerate(bboxes):
        for j, bbox2 in enumerate(bboxes[i+1:], start=i+1):
            if bbox1.intersects(bbox2):
                answer.append(tuple(sorted((i, j))))
    assert len(answer) == len(set(answer))
    tree = AABBTree()
    for i, bbox in enumerate(bboxes):
        assert len(tree) == i
        tree.add(bbox, value=i)
        assert len(tree) == i + 1
    intersections = sorted(
        tuple(sorted(pair)) for pair
        in tree.get_all_intersections()
    )
    assert len(intersections) == len(answer), (intersections, answer)
    assert set(intersections) == set(answer), (intersections, answer)


@given(strats.lists(bounding_boxes()))
def test_aabb_tree_remove(bboxes: list[BoundingBox]) -> None:
    answer = []
    for i, bbox1 in enumerate(bboxes):
        for j, bbox2 in enumerate(bboxes[i+1:], start=i+1):
            if bbox1.intersects(bbox2):
                answer.append(tuple(sorted((i, j))))
    tree = AABBTree()
    for i, bbox in enumerate(bboxes):
        tree.add(bbox, value=i)
    for i, bbox in enumerate(bboxes):
        assert len(tree) == len(bboxes) - i
        tree.remove(bbox, value=i)
        assert len(tree) == len(bboxes) - i - 1
        answer = [
            pair for pair in answer
            if i not in pair
        ]
        intersections = sorted(
            tuple(sorted(pair)) for pair
            in tree.get_all_intersections()
        )
        assert len(intersections) == len(answer), (intersections, answer)
        assert set(intersections) == set(answer), (intersections, answer)


@given(strats.lists(bounding_boxes()), bounding_boxes())
def test_aabb_tree_intersects_with(bboxes: list[BoundingBox], target: BoundingBox) -> None:
    answer = set()
    for i, bbox in enumerate(bboxes):
        if target.intersects(bbox):
            answer.add(i)
    tree = AABBTree()
    for i, bbox in enumerate(bboxes):
        tree.add(bbox, value=i)
    intersections = set(tree.get_intersections_with(target))
    assert intersections == answer, (intersections, answer)
