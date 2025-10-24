import numpy as np
from typing import List, Tuple, Union, Dict, Any

# Define a type for a coordinate point (expected to be a list or tuple of floats/ints)
CoordinatePoint = Union[Tuple[float, float], List[float]]

def get_center_of_bbox(bbox):
    # bbox can be [x1, y1, x2, y2], {'x1':..., 'y1':..., 'x2':..., 'y2':...}, or {id: [x1, y1, x2, y2]}
    if isinstance(bbox, dict):
        # Case 1: dict with keys 'x1', 'y1', 'x2', 'y2'
        if all(k in bbox for k in ('x1', 'y1', 'x2', 'y2')):
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        # Case 2: dict with a single key and value is a list [x1, y1, x2, y2]
        elif len(bbox) == 1 and isinstance(next(iter(bbox.values())), (list, tuple)):
            x1, y1, x2, y2 = next(iter(bbox.values()))
        else:
            raise KeyError(f"Unknown bbox dict format: {bbox}")
    else:
        # assume list or tuple [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)

def measure_distance(p1, p2):
    # Support both (x, y) tuples/lists and {'x':..., 'y':...} dicts
    if isinstance(p1, dict):
        x1, y1 = p1['x'], p1['y']
    else:
        x1, y1 = p1
    if isinstance(p2, dict):
        x2, y2 = p2['x'], p2['y']
    else:
        x2, y2 = p2

    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def get_height_of_bbox(bbox):
    return bbox[3]-bbox[1]

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

def measure_xy_distance(p1: CoordinatePoint, p2: CoordinatePoint) -> Tuple[float, float]:
    """
    Calculates the distance in X and Y between two points.
    Assumes points are in the standardized format: [x, y].

    Args:
        p1 (CoordinatePoint): The first point [x1, y1].
        p2 (CoordinatePoint): The second point [x2, y2].

    Returns:
        Tuple[float, float]: The difference in (x, y) coordinates (p1 - p2).
    """
    try:
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])

        # Calculate the difference (p1 - p2)
        return x1 - x2, y1 - y2
    except (TypeError, IndexError, ValueError) as e:
        print(f"Error in measure_xy_distance. Check if points are in [x, y] format: {p1}, {p2}. Error: {e}")
        raise


def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    """
    Find the closest keypoint (by vertical distance) to the given point.
    keypoints: dict[kp_id] -> (x, y)
    keypoint_indices: list of candidate kp_id values
    """
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]
    for keypoint_indix in keypoint_indices:
        if keypoint_indix not in keypoints:
            continue
        keypoint = keypoints[keypoint_indix]
        distance = abs(point[1] - keypoint[1])  # vertical (y) distance only

        if distance < closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_indix

    return key_point_ind