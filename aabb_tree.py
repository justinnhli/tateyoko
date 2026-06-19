"""An Axis-Aligned Bounding-Box (AABB) tree."""

from dataclasses import dataclass
from itertools import combinations
from functools import cached_property
from math import inf as INF
from typing import Any, Iterator # pylint: disable = unused-import


@dataclass(frozen=True, order=True)
class BoundingBox:
    """An axis-aligned bounding box."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __iter__(self)-> Iterator[float]:
        return iter((self.min_x, self.min_y, self.max_x, self.max_y))

    def __contains__(self, other: BoundingBox) -> bool:
        self._validate()
        return (
            other.min_x >= self.min_x
            and other.min_y >= self.min_y
            and other.max_x <= self.max_x
            and other.max_y <= self.max_y
        )

    def __repr__(self) -> str:
        return f'BoundingBox(({self.min_x}, {self.min_y}), ({self.max_x}, {self.max_y}))'

    def _validate(self) -> None:
        assert self.min_x <= self.max_x, (self.min_x, self.max_x)
        assert self.min_y <= self.max_y, (self.min_y, self.max_y)

    @cached_property
    def area(self):
        # type: () -> float
        """The area of the bounding box."""
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)

    def union(self, other):
        # type: (BoundingBox) -> BoundingBox
        """Create a new bounding box that encompasses both boxes."""
        self._validate()
        result = BoundingBox(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y),
        )
        assert self in result, (self, result)
        assert other in result, (other, result)
        return result

    def intersection(self, other):
        # type: (BoundingBox) -> BoundingBox
        """Calculate the intersection of the two bounding boxes."""
        min_x = max(self.min_x, other.min_x)
        min_y = max(self.min_y, other.min_y)
        max_x = min(self.max_x, other.max_x)
        max_y = min(self.max_y, other.max_y)
        if max_x < min_x or max_y < min_y:
            return None
        else:
            return BoundingBox(min_x, min_y, max_x, max_y)

    def intersects(self, other, include_border=True):
        # type: (BoundingBox, bool) -> bool
        """Determine if two bounding boxes intersect."""
        return (
            (
                self.min_x <= other.min_x < self.max_x
                or (include_border and other.min_x == self.max_x)
                or other.min_x <= self.min_x < other.max_x
                or (include_border and self.min_x == other.max_x)
            ) and (
                self.min_y <= other.min_y < self.max_y
                or (include_border and other.min_y == self.max_y)
                or other.min_y <= self.min_y < other.max_y
                or (include_border and self.min_y == other.max_y)
            )
        )


class AABBNode:
    """A node in an AABB tree."""

    NUM_CHILDREN = 2

    def __init__(self, bounding_box, value=None, children=None):
        # type: (BoundingBox, Any, tuple[AABBNode, ...]) -> None
        self.bounding_box = bounding_box
        self.value = value
        if children is None:
            self.is_leaf = True
            self._children = AABBNode.NUM_CHILDREN * (None,) # type: tuple[AABBNode, ...]
            self.used = self.bounding_box.area
            self.size = 1
        else:
            self.is_leaf = False
            self._set_children(children)

    @property
    def utilization(self):
        # type: () -> float
        """Calculate the utilization rate of the node."""
        if self.bounding_box.area == 0:
            return 0
        else:
            return self.used / self.bounding_box.area

    @property
    def children(self):
        # type: () -> tuple[AABBNode, ...]
        """Get the children of this node."""
        return self._children

    @children.setter
    def children(self, children):
        # type: (tuple[AABBNode, ...]) -> None
        self._set_children(children)

    def _set_children(self, children):
        # type: (tuple[AABBNode, ...]) -> None
        if len(children) != AABBNode.NUM_CHILDREN:
            raise ValueError(f'invalid number of children; expected {AABBNode.NUM_CHILDREN} but got {len(children)}')
        self._children = children
        self.used = sum(child.used for child in self._children)
        self.size = sum(child.size for child in self._children)
        self.bounding_box = self._children[0].bounding_box
        for child in self._children[1:]:
            self.bounding_box = self.bounding_box.union(child.bounding_box)

    def set_child(self, index, child):
        # type: (int, AABBNode) -> None
        """Set a specific child of this node."""
        if not (0 <= index < AABBNode.NUM_CHILDREN):
            raise ValueError(f'invalid index: {index}')
        self.children = (
            *self._children[:index],
            child,
            *self._children[index+1:],
        )


class AABBTree:
    """An axis-aligned bounding-box (AABB) tree."""

    def __init__(self):
        # type: () -> None
        self.root = None # type: AABBNode
        self.size = 0

    def __len__(self):
        # type: () -> int
        return self.size

    def _best_child(self, bounding_box, node):
        # use (-utilization, area) as the heuristic priority
        min_index = 0
        min_priority = (INF, INF)
        priority: tuple[float, float]
        for i, child in enumerate(node.children):
            new_box = child.bounding_box.union(bounding_box)
            if new_box.area == 0:
                priority = (-INF, 0)
            else:
                priority = (
                    -(child.used + bounding_box.area) / new_box.area,
                    new_box.area,
                )
            if priority < min_priority:
                min_index = i
                min_priority = priority
        return min_index

    def _optimize_node(self, node):
        # type: (AABBNode) -> AABBNode
        """Optimize this node heuristically.

        The goal here is to make the tree more efficient by considering
        all arrangements of descendants up to depth d. d = 1 is trivial,
        since there is only one way to arrange two children, but d = 2
        (and therefore up to 2^2 = 4 subtrees) has 14:
        
        * 2 cases where all four subtrees are at the same depth
        * 12 cases where the four subtrees form a degenerate tree
        
        As the recursion pops back up to the root, at each node, consider
        which of these arrangements has the "best" split. Parallels could
        be drawn with how binary search trees "rotate" on the way up after
        insertion/removal. For AABB trees, this is still only a heuristic,
        but as depth d increases, the tree will be increasingly optimized,
        at the cost of exponential computation time (in the extreme case,
        a large enough d would encompass all nodes in a subtree).
        """
        return node # TODO

    def add(self, bounding_box, value=None):
        # type: (BoundingBox, Any) -> None
        """Add a bounding box (and optional associated value) to the tree."""
        if self.root is None:
            self.root = AABBNode(bounding_box, value)
        else:
            self._add_iterative(bounding_box, value)
        self.size += 1

    def _add_iterative(self, bounding_box, value):
        # type: (BoundingBox, Any) -> None
        # do this iteratively to avoid recursion depth limits
        # initialize the stack with a dummy parent of the root
        stack = [(
            AABBNode(
                BoundingBox(0, 0, 0, 0),
                children=(self.root, self.root),
            ),
            0,
        )]
        # recurse down to the leaf
        while True:
            parent, child_index = stack[-1]
            node = parent.children[child_index]
            if node.is_leaf:
                break
            stack.append((node, self._best_child(bounding_box, node)))
        # create the new node (FIXME assumes two children)
        new_node = AABBNode(
            node.bounding_box.union(bounding_box),
            children=(
                node,
                AABBNode(bounding_box, value),
            ),
        )
        # pop back up the stack, setting the children along the way
        for node, child_index in reversed(stack):
            node.set_child(child_index, new_node)
            new_node = self._optimize_node(node)
        self.root = new_node.children[0]

    def remove(self, bounding_box, value=None):
        # type: (BoundingBox, Any) -> None
        """Remove a bounding box from the tree.

        This function is iterative to avoid recursion depth limits.
        """
        if self.root is None:
            raise ValueError(f'value not in tree: {value}')
        # need a variable to ensure only the first matching element is removed
        removed = False
        # initialize the stack with the root
        stack: list[tuple[AABBNode, list[AABBNode]]] = [(
            # parent
            self.root,
            # new children
            [],
        )]
        returned = False
        return_value = None
        # manage our own stack in the "recursion"
        while stack:
            parent, new_children = stack[-1]
            index = len(new_children)
            if parent.is_leaf:
                # base case
                if not removed and parent.bounding_box == bounding_box and parent.value == value:
                    self.size -= 1
                    return_value = None
                    removed = True
                else:
                    return_value = parent
                returned = True
                stack.pop(-1)
            else:
                # if there is a return value, add it as a child
                if returned:
                    new_children.append(return_value)
                    return_value = None
                    returned = False
                    index += 1
                if index == len(parent.children):
                    # if all the new children are there, pop up the stack
                    # FIXME assumes two children
                    if new_children[0] is None:
                        return_value = new_children[1]
                    elif new_children[1] is None:
                        return_value = new_children[0]
                    else:
                        parent.children = new_children
                        return_value = parent
                    returned = True
                    stack.pop(-1)
                else:
                    # if not, continue with its children
                    child = parent.children[index]
                    if not removed and child.bounding_box.intersects(bounding_box):
                        stack.append((child, []))
                    else:
                        new_children.append(child)
        self.root = return_value
        if not removed:
            raise ValueError(f'value not in tree: {value}')

    def get_all_intersections(self):
        # type: () -> Iterator[tuple[Any, Any]]
        """Get all intersecting bounding boxes currently in the tree."""
        if len(self) < 2:
            return
        for child1, child2 in combinations(self.root.children, 2):
            yield from self._get_all_intersections(child1, child2, siblings=True)

    def _get_all_intersections(self, node1, node2, siblings=False):
        # type: (AABBNode, AABBNode, bool) -> Iterator[tuple[Any, Any]]
        # recursive case: recurse on node1's children
        if not node1.is_leaf and siblings:
            for child1, child2 in combinations(node1.children, 2):
                yield from self._get_all_intersections(child1, child2, siblings=True)
        # recursive case: recurse on node2's children
        if not node2.is_leaf and siblings:
            for child1, child2 in combinations(node2.children, 2):
                yield from self._get_all_intersections(child1, child2, siblings=True)
        # base case: not intersecting
        if not node1.bounding_box.intersects(node2.bounding_box):
            return
        # nodes intersect
        if node1.is_leaf and node2.is_leaf:
            # base case: both nodes are leaves; yield their values
            yield node1.value, node2.value
        elif node1.is_leaf:
            # recursive case: node1 is a leaf; recurse on it and node2's children
            for child in node2.children:
                yield from self._get_all_intersections(node1, child)
        elif node2.is_leaf:
            # recursive case: node2 is a leaf; recurse on it and node1's children
            for child in node1.children:
                yield from self._get_all_intersections(child, node2)
        else:
            # recursive case: both nodes are trees; recurse on all combinations of children
            for child1 in node1.children:
                for child2 in node2.children:
                    yield from self._get_all_intersections(child1, child2)

    def get_intersections_with(self, bounding_box):
        # type: (BoundingBox) -> Iterator[Any]
        """Get all bounding boxes that intersect with the given bounding box."""
        if self.root is None:
            return
        frontier = [self.root]
        while frontier:
            node = frontier.pop(0)
            if not node.bounding_box.intersects(bounding_box):
                continue
            if node.is_leaf:
                yield node.value
                continue
            frontier.extend(node.children)

    def pprint(self):
        # type: () -> None
        """Pretty print the tree."""
        self._pprint(self.root)

    def _pprint(self, node, depth=0):
        # type: (AABBNode, int) -> None
        if node is None:
            return
        print(''.join([
            depth * '  ' + f'{node.size}',
            f'({node.bounding_box}; {node.used} / {node.bounding_box.area} = {node.utilization})',
        ]))
        #print(depth * '  ' + f'{node.size} ({node.bounding_box}; used={node.used})')
        if node.value is None:
            for child in node.children:
                self._pprint(child, depth + 1)
